import logging
from KG_LFM.evaluator import KGLFMEvaluator, compute_hit_k
import torch
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple, Sequence, Iterable
from collections import OrderedDict
from dataclasses import dataclass
import networkx as nx
from torch_geometric.data import Data, Batch
import spacy
from spacy.matcher import Matcher
import json
import re

# Prefer rapidfuzz if available (faster than fuzzywuzzy)
try:  # pragma: no cover
    from rapidfuzz import fuzz, process  # type: ignore

    def _fuzzy_topk(q: str, choices: Iterable[str], limit: int = 3):
        return list(process.extract(q, choices, scorer=fuzz.ratio, limit=limit))

except Exception:  # pragma: no cover
    from fuzzywuzzy import fuzz, process  # type: ignore

    def _fuzzy_topk(q: str, choices: Iterable[str], limit: int = 3):
        return list(process.extract(q, choices, scorer=fuzz.ratio, limit=limit))

from KG_LFM.utils.Datasets.factories.factory import trex_star_graphs_factory
from KG_LFM.utils.BigGraphNodeEmb import BigGraphAligner
from KG_LFM.configuration import SPECIAL_KG_TOKEN

# Tool schema for function calling
KG_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_knowledge_graph",
        "description": (
            "Query the knowledge graph for information about an entity. ALWAYS use this when you need "
            "factual information about people, places, organizations, or other entities. "
            "For complex questions, you may need to use this tool multiple times to chain together facts. "
            "It returns inside <concepts> tags the factual vectors coming from the knowledge graph."
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
            self._add_label(eid, eid)
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
            "You are a helpful assistant designed to answer user queries. Please think step-by-step. "
            "Use the `query_knowledge_graph` tool to find factual information. For complex questions, "
            "you may need to use this tool multiple times to build a chain of reasoning by using the "
            "retrieved knowledge in subsequent queries. Once you have gathered all the necessary "
            "information, provide the final answer."
        )
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized KGLFMThinkingEvaluator.")

        # From KGChatBot
        self.max_new_tokens = 256  # Default, can be configured
        self.temperature = 0.7     # Default, can be configured
        self.top_p = 0.9           # Default, can be configured

    def load_model(self):
        super().load_model()
        self.logger.info("Loading knowledge graphs for thinking evaluator...")
        self.entity_graphs = trex_star_graphs_factory(self.config.dataset)
        self.graph_aligner = BigGraphAligner(graphs=self.entity_graphs, config=self.config.dataset)
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

    def _get_dummy_graph_data(self) -> Data:
        """Create a dummy graph data with correct embedding dimensions."""
        try:
            if self.entity_graphs:
                sample_eid = next(iter(self.entity_graphs.keys()))
                emb_dim = int(self.graph_aligner.node_embedding(sample_eid).shape[-1])
            else:
                emb_dim = 768  # fallback
        except Exception:
            emb_dim = 768  # fallback
        dummy_emb = torch.zeros(1, emb_dim)
        return Data(
            x=dummy_emb,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, emb_dim)),
            num_nodes=1,
        )

    def _get_graph_batch(self, entity_ids: Sequence[str]) -> Optional[Batch]:
        if not entity_ids:
            return None
        datas: List[Data] = []
        for eid in entity_ids:
            g = self.entity_graphs.get(eid)
            if g is None:
                self.logger.warning(f"Graph not found for entity {eid}, using dummy embedding")
                datas.append(self._get_dummy_graph_data())
                continue
            if eid in self._graph_cache:
                datas.append(self._graph_cache[eid])
                self._graph_cache.move_to_end(eid)
                continue
            d = self._networkx_to_pyg(g, eid)
            if d is None:
                self.logger.warning(f"Graph conversion failed for entity {eid}, using dummy embedding")
                datas.append(self._get_dummy_graph_data())
                continue
            self._graph_cache[eid] = d
            self._graph_cache.move_to_end(eid)
            datas.append(d)
            while len(self._graph_cache) > 64:
                self._graph_cache.popitem(last=False)
        return Batch.from_data_list(datas) if datas else None

    # ---------- Prompt construction (from KGChatBot) ----------
    def _apply_chat_template(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        tok = self.tokenizer
        try:
            # Ensure `add_generation_prompt` is True for inference
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools if tools else None,
            )
        except Exception:
            # Fallback from inference.py
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

    def _generate_with_thinking(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Perform the multi-step thinking process for a batch and return final logits,
        input_ids, and object boundaries, using the ground truth for the final answer.
        """
        batch_size = len(batch['sentences'])
        final_logits = []
        final_input_ids = []
        final_object_boundaries = []

        for i in range(batch_size):
            full_conversation = batch['conversations'][i]
            # The user query is the first message
            user_query = full_conversation[0]
            # The ground truth answer is the last message
            ground_truth_answer = full_conversation[-1]

            # Start the conversation with the system prompt and the user query
            conversation = [self.SYSTEM_PROMPT, user_query]
            all_entity_ids = []
            
            max_depth = 5
            for _ in range(max_depth):
                prompt = self._apply_chat_template(conversation, tools=[KG_QUERY_TOOL])
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(self.accelerator.device)

                model_on_device = self.accelerator.unwrap_model(self.model)
                generated_outputs = model_on_device.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated_text = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)[0]
                
                tool_calls = self._parse_tool_calls(generated_text)
                if not tool_calls:
                    # Model has finished thinking, break the loop
                    break

                fn_text, new_entity_ids = self._handle_tool_calls(tool_calls)
                all_entity_ids.extend(new_entity_ids)
                
                # Add the assistant's thinking and the tool's response to the conversation
                conversation.append({"role": "assistant", "content": generated_text})
                conversation.append({"role": "tool", "content": fn_text})

            # Now, append the ground truth answer to the conversation
            conversation.append(ground_truth_answer)
            
            # Create the final prompt with the full thinking process and ground truth answer
            final_prompt = self._apply_chat_template(conversation, add_generation_prompt=False)
            
            # Find the object boundaries in the ground truth answer part of the final prompt
            obj_str = batch["sentences"][i][batch["objects"][i]["boundaries"][0]:batch["objects"][i]["boundaries"][1]]
            new_obj_start_char = final_prompt.rfind(obj_str)
            
            if new_obj_start_char == -1:
                self.logger.warning(f"Object string not found in the final prompt for sample {i}. Skipping.")
                continue

            new_obj_end_char = new_obj_start_char + len(obj_str)
            final_tokenized = self.tokenizer(final_prompt, return_tensors='pt').to(self.accelerator.device)
            new_tok_start = final_tokenized.char_to_token(0, new_obj_start_char)
            new_tok_end = final_tokenized.char_to_token(0, new_obj_end_char)

            if new_tok_start is None or new_tok_end is None:
                self.logger.warning(f"Could not find token boundaries for object in sample {i}. Skipping.")
                continue

            graphs = self._get_graph_batch(all_entity_ids)
            
            model_input = {
                'input_ids': final_tokenized['input_ids'],
                'attention_mask': final_tokenized['attention_mask'],
                'graphs': graphs.to(self.accelerator.device) if graphs else None
            }
            
            model_output = model_on_device(**model_input)
            
            final_logits.append(model_output.logits)
            final_input_ids.append(final_tokenized['input_ids'])
            final_object_boundaries.append((new_tok_start, new_tok_end))

        if not final_logits:
            return None, None, None

        max_len_logits = max(l.size(1) for l in final_logits)
        max_len_inputs = max(i.size(1) for i in final_input_ids)
        
        padded_logits = []
        padded_input_ids = []

        for i in range(len(final_logits)):
            logit_pad_len = max_len_logits - final_logits[i].size(1)
            padded_l = torch.nn.functional.pad(final_logits[i], (0, 0, 0, logit_pad_len), 'constant', 0)
            padded_logits.append(padded_l)

            input_pad_len = max_len_inputs - final_input_ids[i].size(1)
            padded_i = torch.nn.functional.pad(final_input_ids[i], (0, input_pad_len), 'constant', self.tokenizer.pad_token_id)
            padded_input_ids.append(padded_i)

        return torch.cat(padded_logits, dim=0), torch.cat(padded_input_ids, dim=0), final_object_boundaries


    def compute_hit_k_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        if self.accelerator.is_main_process:
            self.logger.info(f"Computing Hit@k metrics with thinking for k={k_values}...")

        results = {}
        for name, (preprocess_func, model) in self.tests.items():
            if name != "KG_LFM":
                continue

            if self.accelerator.is_main_process:
                self.logger.info(f"Evaluating {name} model for Hit@k metrics with thinking...")

            hit_k_correct = {k: 0 for k in k_values}
            total_objects = 0
            average_num_tokens = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Computing Hit@k metrics for {name}", disable=not self.accelerator.is_main_process)):
                    if self.max_samples and (batch_idx * self.batch_size * self.accelerator.num_processes) >= self.max_samples:
                        break
                    
                    logits, input_ids, object_boundaries = self._generate_with_thinking(batch)

                    if logits is None:
                        continue

                    # Create a dummy attention mask for compute_hit_k
                    attention_mask = torch.ones_like(input_ids)

                    hit_k_correct_batch, batch_avg_num_tokens, new_objects = compute_hit_k(
                        logits, input_ids, k_values,
                        object_boundaries, self.model_config.num_quantizers,
                        attention_mask,
                        special_token=self.special_kg_token_id, tokenizer=self.tokenizer
                    )

                    average_num_tokens += batch_avg_num_tokens
                    total_objects += new_objects

                    for k in k_values:
                        hit_k_correct[k] += hit_k_correct_batch[k]

            # Gather results
            hit_k_tensors = {}
            for k in k_values:
                hit_k_tensor = torch.tensor(hit_k_correct[k], device=self.accelerator.device)
                hit_k_tensors[k] = self.accelerator.gather(hit_k_tensor).sum().item()
            
            total_objects_tensor = torch.tensor(total_objects, device=self.accelerator.device)
            total_objects_gathered = self.accelerator.gather(total_objects_tensor).sum().item()
            average_num_tokens_gathered = self.accelerator.gather(torch.tensor(average_num_tokens, device=self.accelerator.device)).sum().item()
            
            self.accelerator.wait_for_everyone()

            metrics = {}
            if self.accelerator.is_main_process:
                if total_objects_gathered > 0:
                    average_num_tokens_gathered /= total_objects_gathered
                    metrics['average_num_tokens'] = average_num_tokens_gathered
                    for k in k_values:
                        metrics[f'hit_at_{k}'] = hit_k_tensors[k] / total_objects_gathered
                    self.logger.info(f"Hit@k computed on {total_objects_gathered} objects.")
                else:
                    self.logger.warning("No valid objects found for Hit@k computation")
                    for k in k_values:
                        metrics[f'hit_at_{k}'] = 0.0
            
            results[name] = metrics
        
        return results
