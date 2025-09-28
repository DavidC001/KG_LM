import logging

import nltk
from KG_LM.evaluator import KGLFMEvaluator, compute_hit_k
import torch
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple, Sequence, Iterable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import networkx as nx
from torch_geometric.data import Data, Batch
import spacy
from spacy.matcher import Matcher
import json
import re
from copy import deepcopy

from fuzzywuzzy import fuzz, process  # type: ignore

def _fuzzy_topk(q: str, choices: Iterable[str], limit: int = 3):
    return list(process.extract(q, choices, scorer=fuzz.ratio, limit=limit))

from KG_LM.utils.Datasets.factories.factory import trex_star_graphs_factory, web_qsp_factory
from KG_LM.utils.BigGraphNodeEmb import BigGraphAligner
from KG_LM.configuration import SPECIAL_KG_TOKEN

from accelerate.utils import broadcast_object_list

# Tool schema for function calling
KG_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_knowledge_graph",
        "description": (
            "Query the knowledge graph for information about an entity. ALWAYS use this when you need "
            "factual information about people, places, organizations, or other entities. "
            "It returns concept vectors (not real words) representing the factual information coming from the knowledge graph."
            "Reason about it but do not repeat all of it"
            "For complex questions, you may need to use this tool multiple times to chain together facts. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "The name of the entity to query in the knowledge graph",
                }
            },
            "required": ["entity_name"],
        },
    },
}

@dataclass(frozen=True)
class EntityMatch:
    mention: str
    entity_id: str
    confidence: float  # 0..1

class EntityRecognizer:
    """NER + fuzzy linking against entities present in our graphs."""

    def __init__(self, entity_graphs: Dict[str, nx.DiGraph], fuzzy_threshold: int = 80) -> None:
        self.entity_graphs = entity_graphs
        self.fuzzy_threshold = fuzzy_threshold
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:  # pragma: no cover
            logging.warning(f"spaCy model not available ({e}); using blank English (no NER)")
            self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)
        self._build_index()
        self._setup_patterns()

    def _build_index(self) -> None:
        self.entity_to_id: Dict[str, str] = {}
        self.entity_labels: List[str] = []
        for eid, g in self.entity_graphs.items():
            self._add_label(eid, eid)  # allow direct id hits
            for _, data in g.nodes(data=True):
                lab = (data.get("label") or "").strip().lower()
                if lab:
                    self._add_label(lab, eid)
        logging.info(f"Indexed {len(self.entity_to_id)} labels for entity linking")

    def _add_label(self, label: str, eid: str) -> None:
        if label not in self.entity_to_id:
            self.entity_to_id[label] = eid
            self.entity_labels.append(label)

    def _setup_patterns(self) -> None:
        # Tiny heuristic: role keywords
        self.matcher.add("ROLE", [[{"LOWER": {"IN": ["president", "ceo", "director", "minister"]}}]])

    def extract(self, text: str, limit: int = 2) -> List[EntityMatch]:
        if not text.strip():
            return []
        doc = self.nlp(text)
        cands: List[str] = []
        if "ents" in dir(doc):
            cands += [ent.text.lower().strip() for ent in doc.ents]
        for _, s, e in self.matcher(doc):
            cands.append(doc[s:e].text.lower().strip())
        if not cands:  # very light fallback
            cands = [t.text.lower() for t in doc if t.is_alpha and len(t) > 2]

        matches: List[EntityMatch] = []
        for c in cands:
            if not c:
                continue
            eid = self.entity_to_id.get(c)
            if eid:
                matches.append(EntityMatch(c, eid, 1.0))
                continue
            for m_text, score, *_ in _fuzzy_topk(c, self.entity_labels, limit=3):
                if score >= self.fuzzy_threshold:
                    matches.append(EntityMatch(c, self.entity_to_id[m_text], score / 100.0))
                    break

        # Deduplicate by (mention,eid) with max confidence
        best: Dict[Tuple[str, str], float] = {}
        for m in matches:
            key = (m.mention, m.entity_id)
            best[key] = max(best.get(key, 0.0), m.confidence)
        out = [EntityMatch(k[0], k[1], c) for k, c in best.items()]
        out.sort(key=lambda x: x.confidence, reverse=True)
        return out[:limit]


class KGLFMThinkingEvaluator(KGLFMEvaluator):
    """
    Evaluator for KG-LFM that incorporates a 'thinking' step, allowing the model
    to query the knowledge graph before generating a final answer.
    """
    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "You are a helpful assistant designed to answer user queries."
            # "Use the `query_knowledge_graph` tool to find factual information. For complex questions, "
            # "you may need to use this tool multiple times to build a chain of reasoning by using the "
            # "retrieved knowledge in subsequent queries. Once you have gathered all the necessary "
            # "information, provide the final answer."
        )
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized KGLFMThinkingEvaluator.")

        # From KGChatBot
        self.max_new_tokens = 512  # Default, can be configured
        self.temperature = 0.7     # Default, can be configured
        self.top_p = 0.9           # Default, can be configured
        
        
        self.evaluations = {
            # 'perplexity': self.compute_perplexity,
            # 'hit_at_k': self.compute_hit_k_metrics,
            "fuzzy_entity_recognition": self.compute_fuzzy_metrics
        }

    def load_model(self):
        super().load_model()
        self.logger.info("Loading knowledge graphs for thinking evaluator...")
        dataset_conf = deepcopy(self.config.dataset)
        _, self.entity_graphs = web_qsp_factory(dataset_conf)
        self.graph_aligner = BigGraphAligner(graphs=self.entity_graphs, config=dataset_conf)
        self.entity_recognizer = EntityRecognizer(self.entity_graphs)
        self._graph_cache: OrderedDict[str, Data] = OrderedDict()

    # ---------- Graph handling (from KGChatBot) ----------
    def _networkx_to_pyg(self, g: nx.DiGraph, central: str) -> Optional[Data]:
        try:
            if central not in g:
                if g.number_of_nodes() == 0:
                    return None
                central = next(iter(g.nodes))
            edges = list(g.out_edges(central, data=True)) or list(g.in_edges(central, data=True))
            if not edges:
                return None
            neigh_ids: List[str] = []
            edge_ids: List[str] = []
            for u, v, ed in edges:
                n = v if u == central else u
                if n == central:
                    continue
                neigh_ids.append(n)
                edge_ids.append(ed.get("id", f"edge::{u}->{v}"))
            try:
                c_emb = self.graph_aligner.node_embedding(central)
                n_emb = self.graph_aligner.node_embedding_batch(neigh_ids)
                e_emb = self.graph_aligner.edge_embedding_batch(edge_ids)
            except Exception as e:
                self.logger.warning(f"Embedding retrieval failed for {central}: {e}")
                return None
            num_n = len(neigh_ids)
            edge_index = torch.tensor(
                [list(range(1, num_n + 1)) + [0] * num_n, [0] * num_n + list(range(1, num_n + 1))],
                dtype=torch.long,
            )
            x = torch.cat([c_emb.unsqueeze(0), n_emb], dim=0)
            edge_attr = torch.cat([e_emb, e_emb], dim=0)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
        except Exception as e:
            self.logger.warning(f"Graph conversion failed for {central}: {e}")
            return None

    def _get_graph_batch(self, entity_ids: Sequence[str]) -> Optional[Batch]:
        if not entity_ids:
            return None
        datas: List[Data] = []
        for eid in entity_ids:
            g = self.entity_graphs.get(eid)
            if g is None:
                raise ValueError(f"Entity ID {eid} not found in graphs")
            if eid in self._graph_cache:
                datas.append(self._graph_cache[eid])
                self._graph_cache.move_to_end(eid)
                continue
            d = self._networkx_to_pyg(g, eid)
            if d is None:
                raise ValueError(f"Entity ID {eid} has no graph data")
            self._graph_cache[eid] = d
            self._graph_cache.move_to_end(eid)
            datas.append(d)
            while len(self._graph_cache) > 64:
                self._graph_cache.popitem(last=False)
        return Batch.from_data_list(datas) if datas else None

    # ---------- Prompt construction (from KGChatBot) ----------
    def _apply_chat_template(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, add_generation_prompt: bool = True) -> str:
        tok = self.tokenizer
        try:
            # Ensure `add_generation_prompt` is True for inference
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                tools=tools if tools else None,
            )
        except Exception:
            # Fallback
            tool_msg = f" (tools: {', '.join([t['function']['name'] for t in tools])})" if tools else ""
            parts = []
            if messages[0]["role"] != "system":
                parts += [f"system: {tool_msg}"]
                parts += [f"{m['role']}: {m['content']}" for m in messages]
            else:
                system_prompt = messages[0]["content"] + "\n" + tool_msg
                parts += [f"system: {system_prompt}"]
                parts += [f"{m['role']}: {m['content']}" for m in messages[1:]]
            parts.append("assistant:")
            return "\n".join(parts)

    # ---------- Tool plumbing (from KGChatBot) ----------
    def query_knowledge_graph(self, entity_name: str) -> Dict[str, Any]:
        """Resolve an entity name to an entity_id present in the graphs."""
        matches = self.entity_recognizer.extract(entity_name, limit=1)
        if not matches:
            return {"ok": False, "error": f"Entity '{entity_name}' not found in knowledge graph"}
        entity_id = matches[0].entity_id
        return {"ok": True, "entity_id": entity_id, "entity_name": entity_name}

    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls emitted as JSON in the text."""
        tool_calls: List[Dict[str, Any]] = []
        pattern = r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}"
        for m in re.findall(pattern, response_text):
            try:
                obj = json.loads(m)
                if isinstance(obj.get("arguments"), str):
                    obj["arguments"] = json.loads(obj["arguments"])
                tool_calls.append(obj)
            except json.JSONDecodeError:
                continue
        return tool_calls

    def _handle_tool_calls(self, calls: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Execute tool calls and return a function-message text + list of entity_ids."""
        fn_text_lines: List[str] = []
        entity_ids: List[str] = []
        for call in calls:
            if call.get("name") != "query_knowledge_graph":
                continue
            args = call.get("arguments", {})
            entity_name = (args or {}).get("entity_name", "")
            if not entity_name:
                fn_text_lines.append("missing entity_name")
                continue
            result = self.query_knowledge_graph(entity_name)
            if not result.get("ok"):
                fn_text_lines.append(f"{entity_name} not found")
                continue
            eid = result["entity_id"]
            entity_ids.append(eid)
            fn_text_lines.append(f"{entity_name}{SPECIAL_KG_TOKEN}")
        return "\n".join(fn_text_lines) + ("\n" if fn_text_lines else ""), entity_ids

    def _find_token_span_from_char_offsets(self, offsets, char_start: int, char_end: int) -> tuple[int, int] | None:
        """
        Map a character span [char_start, char_end) to token indices using a HF fast tokenizer's
        offset_mapping for one sequence. Returns (tok_start, tok_end) inclusive, or None if not found.
        """
        tok_start = tok_end = None
        for i, (s, e) in enumerate(offsets):
            if s <= char_start < e:
                tok_start = i
                break
        if tok_start is None:
            return None
        # char_end is exclusive; align to last token whose end >= char_end
        for j in range(tok_start, len(offsets)):
            s, e = offsets[j]
            if char_end <= e:
                tok_end = j
                break
        if tok_end is None:
            return None
        return tok_start, tok_end

    def _generate_with_thinking_batched(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[tuple[int, int]] | None, List[str] | None]:
        """
        Returns: logits, input_ids, token_spans, final_decoded_texts
        - logits/input_ids/token_spans are for Hit@k
        - final_decoded_texts is the final answer text per sample for fuzzy scoring
        """
        device = self.accelerator.device
        n = len(batch["sentences"])

        conversations: list[list[Dict[str, str]]] = []
        all_entity_ids_per_sample: list[list[str]] = []
        
        for i in range(n):
            conversations.append([self.SYSTEM_PROMPT, batch["conversations"][i][0], batch["conversations"][i][1]])
            all_entity_ids_per_sample.append([batch["subjects"][i]["id"]])

        active_idx: list[int] = list(range(n))
        max_depth = 2
        final_answers = ["" for _ in range(n)]
        
        for _step in range(max_depth):
            if not active_idx:
                break
            
            self.logger.debug(f"Thinking step {_step+1} with {len(active_idx)} active samples")
            
            # build prompts only for active samples
            prompts = [
                self._apply_chat_template(
                    conversations[i], 
                    tools=[KG_QUERY_TOOL] if _step < max_depth - 1 else None
                ) for i in active_idx
            ]
            enc = self.tokenizer(
                prompts, 
                return_tensors="pt", padding=True, truncation=True,
                padding_side="left",
            ).to(self.accelerator.device)

            # graphs only for active samples, flattened in active sample order
            flat_entity_ids: list[str] = []
            for i in active_idx:
                flat_entity_ids.extend(all_entity_ids_per_sample[i])
            graphs = self._get_graph_batch(flat_entity_ids)
            
            gen = self.model.generate(
                **enc,
                graphs=graphs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            self.logger.debug(f"Model outputs at step {_step+1}: {texts}")

            for i, idx in enumerate(active_idx):
                final_answers[idx] = texts[i]

            progressed_any = False
            new_active_idx: list[int] = []

            # write back per active sample; stop samples that emitted no tool calls
            for j, i in enumerate(active_idx):
                calls = self._parse_tool_calls(texts[j])
                fn_text, new_entity_ids = self._handle_tool_calls(calls)

                conversations[i].append({"role": "assistant", "content": texts[j]})
                if fn_text:
                    conversations[i].append({"role": "tool", "content": fn_text})
                    all_entity_ids_per_sample[i].extend(new_entity_ids)

                if calls:
                    progressed_any = True
                    new_active_idx.append(i)

            if not progressed_any:
                break
            active_idx = new_active_idx

        return final_answers

    # -------------------------
    # New: Fuzzy-matching evaluation (string-level) + WebQSP aggregation
    # -------------------------
    def _extract_gold_answers_for_sample(self, batch: Dict[str, Any], i: int) -> List[str]:
        """
        Try to recover the gold answer strings robustly from common fields.
        """
        answers: List[str] = []

        # 1) object label(s) in objects[i]
        obj = batch["objects"][i]
        for key in ("object_label", "label", "answer"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                answers.append(val.strip())

        # 2) sometimes dataset stores 'answers' list at sample-level
        sample_answers = None
        if isinstance(batch.get("answers"), list):
            sample_answers = batch["answers"][i]
        elif isinstance(batch.get("gold_answers"), list):
            sample_answers = batch["gold_answers"][i]
        if sample_answers:
            if isinstance(sample_answers, (list, tuple)):
                answers.extend([str(a).strip() for a in sample_answers if str(a).strip()])
            elif isinstance(sample_answers, str):
                answers.append(sample_answers.strip())

        # 3) sentence CSV style: there might be a direct 'answer' in a per-sentence dict
        if isinstance(batch.get("sentences_meta"), list):
            meta = batch["sentences_meta"][i]
            a = meta.get("answer")
            if isinstance(a, str) and a.strip():
                answers.append(a.strip())

        # De-duplicate and lowercase for matching
        answers = [a for a in {a.strip(): None for a in answers}.keys()]
        return answers

    def grammarTree_parse(self, result, answers):
        lowersetence=result.lower()
        if "I'm sorry" in result:
            return 0
        text = nltk.word_tokenize(lowersetence)
        sentence=nltk.pos_tag(text)
        #grammar = "NP:{<JJ|NN|NNS.*><POS|IN.*><NN|NNS.*>}"
        grammar = r"""
                    NP:{<JJ|NN><POS|IN>?<NN>+}
                    PP:{<NN|NNS|NNP|NNPS>}

                    """
        cp = nltk.RegexpParser(grammar) #生成规则
        result = cp.parse(sentence) #进行分块

        substring=[]
        finalstring= []
        for subtree in result.subtrees():
            if ((subtree.label() == 'NP')|(subtree.label()=='PP')):
                substring.append(subtree)
        for each in substring:
            length=len(each)
            #for i in (0,length-1):
                #print(each[i])
            final = ''
            for i in range(0,length):
                final += each[i][0] + ' '
            finalstring.append(final)
            
        for st in finalstring:
            #st = st[0]
            for ans in answers:
                if fuzz.ratio(ans.lower(), st) > 50:
                    return 1
        return 0

    def compute_fuzzy_metrics(
        self,
        threshold: int = 50,          # fuzzy ratio threshold to count as hit
        k_values: List[int] = [1,3,5] # optional @k reporting on top fuzzy candidates (useful if you produce multiple)
    ) -> Dict[str, float]:
        """
        Generate final answers with the thinking loop and evaluate them by fuzzy string matching
        against gold answers. For WebQSP, aggregate per question (best among linked entities).
        """
        if self.accelerator.is_main_process:
            self.logger.info(f"Computing fuzzy-matching metrics (threshold={threshold}) with thinking...")

        results = {}
        for name, (preprocess_func, model) in self.tests.items():
            if name != "KG_LM":
                continue

            total = 0
            correct = 0

            # WebQSP aggregation
            is_webqsp = (self.config.dataset.name == "web-qsp")
            per_question_best: Dict[Any, float] = defaultdict(float) if is_webqsp else None

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing fuzzy metrics for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break

                    # Run the full thinking pass and get final decoded texts
                    final_texts = self._generate_with_thinking_batched(batch)

                    # Question ids (for WebQSP)
                    question_ids = batch.get('question_ids', None)
                    if question_ids is None:
                        # build stable synthetic ids (like you did)
                        bs = len(final_texts or [])
                        base_id = batch_idx * self.accelerator.num_processes * bs + self.accelerator.process_index * bs
                        question_ids = list(range(base_id, base_id + bs))

                    # For each sample: fuzzy-match generated text vs gold answers
                    for i, gen_text in enumerate(final_texts or []):
                        golds = self._extract_gold_answers_for_sample(batch, i)
                        if not golds:
                            continue
                        
                        # use nltk to extract all NN and PP
                        correct_s = self.grammarTree_parse(gen_text, golds)
                        if is_webqsp:
                            qid = question_ids[i]
                            per_question_best[qid] = max(per_question_best[qid], correct_s)
                        else:
                            total += 1
                            correct += correct_s

            # Gather across processes
            if not is_webqsp:
                total_t = torch.tensor(total, device=self.accelerator.device)
                correct_t = torch.tensor(correct, device=self.accelerator.device)
                total_g = self.accelerator.gather(total_t).sum().item()
                correct_g = self.accelerator.gather(correct_t).sum().item()
                self.accelerator.wait_for_everyone()

                metrics = {}
                if self.accelerator.is_main_process:
                    acc = (correct_g / total_g) if total_g > 0 else 0.0
                    metrics["fuzzy@%d" % threshold] = acc
                    self.logger.info(f"Fuzzy accuracy @ {threshold}: {acc:.4f} over {total_g} samples")
                results[name] = metrics
            else:
                # Broadcast dicts & merge on main
                gathered = []
                for p in range(self.accelerator.num_processes):
                    local = dict(per_question_best) if p == self.accelerator.process_index else {}
                    payload = [local]
                    broadcast_object_list(payload, from_process=p)
                    gathered.append(payload[0])

                if self.accelerator.is_main_process:
                    merged: Dict[Any, float] = {}
                    for d in gathered:
                        for qid, score in d.items():
                            merged[qid] = max(merged.get(qid, 0.0), score)
                    valid_q = len(merged)
                    
                    acc = sum(merged.values()) / valid_q if valid_q > 0 else 0.0
                    metrics = {"fuzzy@%d" % threshold: acc, "questions": valid_q}
                else:
                    metrics = {}
                
                # broadcast final metrics
                blob = [metrics]
                broadcast_object_list(blob, from_process=0)
                results[name] = blob[0]

        return results
