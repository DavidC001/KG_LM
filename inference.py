#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import networkx as nx
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

# Hugging Face
from transformers import AutoTokenizer

# spaCy + simple pattern matching
import spacy
from spacy.matcher import Matcher

from side_by_side import print_side_by_side

# Prefer rapidfuzz if available (faster than fuzzywuzzy)
try:  # pragma: no cover
    from rapidfuzz import fuzz, process  # type: ignore

    def _fuzzy_topk(q: str, choices: Iterable[str], limit: int = 3):
        return list(process.extract(q, choices, scorer=fuzz.ratio, limit=limit))

except Exception:  # pragma: no cover
    from fuzzywuzzy import fuzz, process  # type: ignore

    def _fuzzy_topk(q: str, choices: Iterable[str], limit: int = 3):
        return list(process.extract(q, choices, scorer=fuzz.ratio, limit=limit))

# PyG
from torch_geometric.data import Data, Batch

# Project imports
from KG_LM.model.KG_LM_arch import KG_LM
from KG_LM.utils.Datasets.factories.factory import trex_star_graphs_factory
from KG_LM.utils.BigGraphNodeEmb import BigGraphAligner
from KG_LM.configuration import DatasetConfig, SPECIAL_KG_TOKEN

# ----------------------------
# Tool schema for function calling
# ----------------------------
KG_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_knowledge_graph",
        "description": (
            "Query the knowledge graph for information about an entity. ALWAYS use this when you need "
            "factual information about people, places, organizations, or other entities. "
            "It returns concept vectors (not real words) representing the factual information coming from the knowledge graph."
            "Reason about it but do not repeat all of it"
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

# ----------------------------
# Logging
# ----------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("KG_LM_cli_compare")

# ----------------------------
# Helpers
# ----------------------------

def _select_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            pref = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            pref = "mps"
        else:
            pref = "cpu"
    logger.info(f"Using device: {pref}")
    return torch.device(pref)


def _ensure_tokenizer_on(model: Any, model_or_path: str):
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_or_path)
        setattr(model, "tokenizer", tok)
    # Ensure pad token exists
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tok))
    return tok

# ----------------------------
# Entity recognition for mapping names -> entity IDs present in the graphs
# ----------------------------
@dataclass(frozen=True)
class EntityMatch:
    mention: str
    entity_id: str
    confidence: float  # 0..1
    entity_name: str


class EntityRecognizer:
    """NER + fuzzy linking against entities present in our graphs."""

    def __init__(self, entity_graphs: Dict[str, nx.DiGraph], fuzzy_threshold: int = 90) -> None:
        self.entity_graphs = entity_graphs
        self.fuzzy_threshold = fuzzy_threshold
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:  # pragma: no cover
            logger.warning(f"spaCy model not available ({e}); using blank English (no NER)")
            self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)
        self._build_index()
        self._setup_patterns()

    def _build_index(self) -> None:
        self.entity_to_id: Dict[str, str] = {}
        self.entity_labels: List[str] = []
        for eid, g in self.entity_graphs.items():
            for entid, data in g.nodes(data=True):
                lab = (data.get("label") or "").strip().lower()
                if lab:
                    self._add_label(lab, entid)
                    break  # only first node per graph
        logger.info(f"Indexed {len(self.entity_to_id)} labels for entity linking")

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

        logger.info(f"Entity candidates from text '{text}': {cands}")

        matches: List[EntityMatch] = []
        for c in cands:
            if not c:
                continue
            eid = self.entity_to_id.get(c)
            if eid:
                logger.info(f"Exact match: '{c}' -> '{eid}'")
                matches.append(EntityMatch(c, eid, 1.0, c))
                continue
            fuzzy_results = _fuzzy_topk(c, self.entity_labels, limit=3)
            logger.info(f"Fuzzy search for '{c}': {fuzzy_results}")
            for m_text, score, *_ in fuzzy_results:
                if score >= self.fuzzy_threshold:
                    matched_eid = self.entity_to_id[m_text]
                    logger.info(f"Fuzzy match: '{c}' -> '{m_text}' -> '{matched_eid}' (score: {score})")
                    matches.append(EntityMatch(c, matched_eid, score / 100.0, m_text))
                    break

        # Deduplicate by (mention,eid) with max confidence
        best: Dict[Tuple[str, str, str], float] = {}
        for m in matches:
            key = (m.mention, m.entity_id, m.entity_name)
            best[key] = max(best.get(key, 0.0), m.confidence)
        out = [EntityMatch(k[0], k[1], c, k[2]) for k, c in best.items()]
        out.sort(key=lambda x: x.confidence, reverse=True)
        return out[:limit]

# ----------------------------
# ChatBot (tools-only + baseline compare)
# ----------------------------
class KGChatBot:
    def __init__(
        self,
        model_path: str,
        dataset_config: DatasetConfig,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "auto",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.device = _select_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Two separate histories to keep the comparison fair
        self.hist_baseline: List[Dict[str, Any]] = []
        self.hist_augmented: List[Dict[str, Any]] = []

        if system_prompt:
            sp = {"role": "system", "content": system_prompt, "kg_entity_ids": []}
            self.hist_baseline.append(dict(sp))
            self.hist_augmented.append(dict(sp))

        logger.info("Loading KG-LFM model…")
        self.model = KG_LM.from_pretrained(model_path)
        self.model.to(self.device).eval()
        self.tokenizer = _ensure_tokenizer_on(self.model, model_path)

        logger.info("Loading knowledge graphs…")
        dataset_config.name = "trirex"
        self.entity_graphs = trex_star_graphs_factory(dataset_config)
        self.graph_aligner = BigGraphAligner(graphs=self.entity_graphs, config=dataset_config)
        self.entity_recognizer = EntityRecognizer(self.entity_graphs)

        self._graph_cache: OrderedDict[str, Data] = OrderedDict()
        self.graph_token_entity_ids: List[str] = []  # augmented path only

    # ---------- Graph handling ----------
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
            except Exception as e:  # pragma: no cover
                logger.warning(f"Embedding retrieval failed for {central}: {e}")
                return None
            num_n = len(neigh_ids)
            edge_index = torch.tensor(
                [list(range(1, num_n + 1)) + [0] * num_n, [0] * num_n + list(range(1, num_n + 1))],
                dtype=torch.long,
            )
            x = torch.cat([c_emb.unsqueeze(0), n_emb], dim=0)
            edge_attr = torch.cat([e_emb, e_emb], dim=0)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
        except Exception as e:  # pragma: no cover
            logger.warning(f"Graph conversion failed for {central}: {e}")
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

    def visualize_graph(self, entity_id: str, max_nodes: int = 40) -> Optional[str]:
        g = self.entity_graphs.get(entity_id)
        if g is None:
            logger.warning(f"No graph found for entity_id: {entity_id}")
            return None
        try:
            # Debug: log graph info
            logger.info(f"Visualizing graph for entity_id: {entity_id}")
            logger.info(f"Graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
            
            # Log some node labels to see what's in this graph
            node_labels = []
            for node, data in list(g.nodes(data=True))[:5]:  # First 5 nodes
                label = data.get("label", node)
                node_labels.append(f"{node}: {label}")
            logger.info(f"Sample nodes: {node_labels}")
            
            # Ensure entity_id is actually in the graph
            if entity_id not in g.nodes:
                logger.warning(f"Entity {entity_id} not found in its own graph nodes")
                # Try to find a node with a matching label
                found_node = None
                for node, data in g.nodes(data=True):
                    if data.get("label", "").lower() == entity_id.lower():
                        found_node = node
                        break
                if found_node:
                    logger.info(f"Found matching node by label: {found_node}")
                    entity_id = found_node
                else:
                    logger.warning(f"Could not find any matching node for {entity_id}")
                    return None
                
            if g.number_of_nodes() > max_nodes:
                keep = {entity_id}
                for u, v in g.edges():
                    keep.add(u)
                    keep.add(v)
                    if len(keep) >= max_nodes:
                        break
                sg = g.subgraph(keep).copy()
            else:
                sg = g
                
            # Ensure entity_id is still in the subgraph
            if entity_id not in sg.nodes:
                logger.warning(f"Entity {entity_id} not in subgraph after filtering")
                return None
                
            plt.figure(figsize=(5, 5))
            try:
                # Use spring layout but ensure central node is at center
                pos = nx.spring_layout(sg, seed=42)
                # Force the central entity to be at (0, 0)
                if entity_id in pos:
                    center_pos = pos[entity_id]
                    # Shift all positions so central entity is at origin
                    for node in pos:
                        pos[node] = (pos[node][0] - center_pos[0], pos[node][1] - center_pos[1])
            except Exception:  # pragma: no cover
                pos = nx.circular_layout(sg)
                
            labels = {n: (d.get("label") or str(n))[:40] for n, d in sg.nodes(data=True)}
            node_colors = ["#ffcc00" if n == entity_id else "#1f78b4" for n in sg.nodes]
            node_sizes = [800 if n == entity_id else 400 for n in sg.nodes]
            nx.draw_networkx_nodes(
                sg, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, linewidths=0.5, edgecolors="black"
            )
            nx.draw_networkx_labels(sg, pos, labels=labels, font_size=8)
            nx.draw_networkx_edges(sg, pos, arrows=True, width=1, alpha=0.6, edge_color="#555555")
            e_labels = {(u, v): (d.get("label") or "")[:25] for u, v, d in sg.edges(data=True)}
            if e_labels:
                nx.draw_networkx_edge_labels(sg, pos, edge_labels=e_labels, font_size=6, label_pos=0.5)
            plt.axis("off")
            
            # Create graphs directory if it doesn't exist
            import os
            graphs_dir = "graphs"
            os.makedirs(graphs_dir, exist_ok=True)
            
            # Generate filename with entity_id and timestamp for uniqueness
            import time
            timestamp = int(time.time())
            safe_entity_id = entity_id.replace("/", "_").replace(":", "_")
            filename = f"graph_{safe_entity_id}_{timestamp}.png"
            filepath = os.path.join(graphs_dir, filename)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=140)
            plt.close()
            return filepath
        except Exception as e:  # pragma: no cover
            logger.warning(f"Graph visualization failed for {entity_id}: {e}")
            return None

    # ---------- Prompt construction ----------
    def _max_input_len(self) -> int:
        cfg = getattr(self.model, "config", None)
        max_pos = getattr(cfg, "max_position_embeddings", None)
        base = 2048 if max_pos is None else int(max_pos)
        # budget a little for generation + special tokens
        return max(256, base - self.max_new_tokens - 16)

    def _apply_chat_template(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        tok = self.tokenizer
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools if tools else None,
            )
        except Exception:
            # Very simple fallback
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

    # ---------- Tool plumbing ----------
    def query_knowledge_graph(self, entity_name: str) -> Dict[str, Any]:
        """Resolve an entity name to an entity_id present in the graphs.
        Returns a structured payload for the tool result.
        """
        # First try the existing entity recognizer
        matches = self.entity_recognizer.extract(entity_name, limit=1)
        
        if not matches:
            return {"ok": False, "error": f"Entity '{entity_name}' not found in knowledge graph"}
        
        entity_id = matches[0].entity_id
        entity_name = matches[0].entity_name
        logger.info(f"Resolved '{entity_name}' -> '{entity_id}' (confidence: {matches[0].confidence:.2f})")
        
        # Double-check that this entity_id actually has a graph
        if entity_id not in self.entity_graphs:
            logger.warning(f"Entity ID '{entity_id}' resolved but no graph found!")
            return {"ok": False, "error": f"Entity '{entity_name}' resolved to '{entity_id}' but no graph available"}
            
        return {"ok": True, "entity_id": entity_id, "entity_name": entity_name}

    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls emitted as JSON in the text (model-dependent)."""
        tool_calls: List[Dict[str, Any]] = []
        # Pattern for objects like {"name":"query_knowledge_graph","arguments":{...}}
        import re

        pattern = r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}"
        for m in re.findall(pattern, response_text):
            try:
                obj = json.loads(m)
                if isinstance(obj.get("arguments"), str):
                    obj["arguments"] = json.loads(obj["arguments"])  # some models double-encode
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
            # Expose a KG token so the model knows graphs will follow
            fn_text_lines.append(f"Found entity {result['entity_name']}{SPECIAL_KG_TOKEN}")
        return "\n".join(fn_text_lines) + ("\n" if fn_text_lines else ""), entity_ids

    # ---------- Generation ----------
    def _attempt_generate(self, inputs: Dict[str, torch.Tensor], graphs: Optional[Batch] = None) -> torch.Tensor:
        gen_kwargs: Dict[str, Any] = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if graphs is not None:
            gen_kwargs["graphs"] = graphs
        try:
            return self.model.generate(**gen_kwargs)
        except RuntimeError as e:  # pragma: no cover
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM; retrying with reduced max_new_tokens")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gen_kwargs["max_new_tokens"] = max(32, self.max_new_tokens // 2)
                return self.model.generate(**gen_kwargs)
            raise

    # ---------- Public compare API ----------
    def respond_compare(self, user_input: str) -> Tuple[str, str, List[Tuple[str, str, float]]]:
        """Return (baseline_text, augmented_text, resolved_entities)."""
        # 1) Append user to both histories
        u_msg = {"role": "user", "content": user_input}
        self.hist_baseline.append(dict(u_msg))
        self.hist_augmented.append(dict(u_msg))

        max_inp = self._max_input_len()

        # 2) BASELINE (no tools, no graphs)
        baseline_prompt = self._apply_chat_template(self.hist_baseline, tools=None)
        baseline_inputs = self.tokenizer(
            baseline_prompt, return_tensors="pt", truncation=True, max_length=max_inp
        ).to(self.device)
        
        with torch.inference_mode():
            # check if llm has method disable_adapter
            if hasattr(self.model.llm, "disable_adapter"):
                with self.model.llm.disable_adapter():
                    g_baseline = self._attempt_generate(baseline_inputs, graphs=None)
            else:
                g_baseline = self._attempt_generate(baseline_inputs, graphs=None)

        # The model's generate method now returns only newly generated tokens
        baseline_text = self.tokenizer.decode(g_baseline[0], skip_special_tokens=True).strip()
        self.hist_baseline.append({"role": "assistant", "content": baseline_text})

        # 3) AUGMENTED (tools-only)
        # 3a) First pass: allow tool calling
        aug_prompt = self._apply_chat_template(self.hist_augmented, tools=[KG_QUERY_TOOL])
        aug_inputs = self.tokenizer(aug_prompt, return_tensors="pt", truncation=True, max_length=max_inp).to(self.device)
        with torch.inference_mode():
            g_first = self._attempt_generate(aug_inputs, graphs=self._get_graph_batch(self.graph_token_entity_ids))
        # The model's generate method now returns only newly generated tokens
        first_text = self.tokenizer.decode(g_first[0], skip_special_tokens=True).strip()
        self.hist_augmented.append({"role": "assistant", "content": first_text})

        tool_calls = self._parse_tool_calls(first_text)
        
        max_depth = 5
        aug_res = first_text
        resolved: List[Tuple[str, str, float]] = []
        while len(tool_calls)>0 and max_depth > 0:
            # 3b) Parse + execute tool calls
            fn_text, new_entity_ids = self._handle_tool_calls(tool_calls)
            
            if new_entity_ids:
                # Record entities as (mention?, id, confidence=1.0) — we don't store mention text here
                resolved = [("tool_query", eid, 1.0) for eid in new_entity_ids]
                self.graph_token_entity_ids.extend(new_entity_ids)
            
            # 3c) Add function result and produce final augmented answer
            self.hist_augmented.append({
                "role": 'tool' if "tool" in self.tokenizer.chat_template else 'assistant', 
                "content": fn_text
            })
            aug_prompt2 = self._apply_chat_template(self.hist_augmented, tools=[KG_QUERY_TOOL])
            aug_inputs2 = self.tokenizer(aug_prompt2, return_tensors="pt", truncation=True, max_length=max_inp).to(self.device)
            graphs = self._get_graph_batch(self.graph_token_entity_ids)
            graphs = graphs.to(self.device) if graphs is not None else None
            with torch.inference_mode():
                g_second = self._attempt_generate(aug_inputs2, graphs=graphs)
                
            augmented_text = self.tokenizer.decode(g_second[0], skip_special_tokens=True).strip()
            self.hist_augmented.append({"role": "assistant", "content": augmented_text})
            aug_res += "\n====\n" + fn_text + "\n====\n" + augmented_text

            tool_calls = self._parse_tool_calls(augmented_text)
            max_depth -= 1

        return baseline_text, aug_res, resolved

# ----------------------------
# CLI runner
# ----------------------------

def _print_entities(ents: List[Tuple[str, str, float]]):
    if not ents:
        print("No entities resolved (augmented path).")
        return
    print("Resolved entities (augmented):")
    for i, (m, eid, conf) in enumerate(ents, 1):
        print(f"  [{i}] {m} -> {eid} ({conf:.0%})")


def run_cli(bot: KGChatBot) -> None:
    print("KG-LFM CLI (Compare Mode • Tools-Only) — type :help for commands")
    last_ents: List[Tuple[str, str, float]] = []

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Exiting.")
            break
        if not user:
            continue
        low = user.lower()
        if low in {":quit", ":exit", "q"}:
            break
        if low == ":help":
            print(
                (
                    "\nCommands:\n"
                    "  :help                 Show this help\n"
                    "  :quit / :exit / q     Exit\n"
                    "  :entities             Show last resolved entities (augmented path)\n"
                    "  :graph <id|index>     Save graph image for an entity id or 1-based index\n\n"
                    "About compare mode:\n"
                    "  - BASELINE is generated with no tools and no graphs.\n"
                    "  - KG-AUGMENTED allows tool calls to query the KG and then attaches graphs.\n"
                ).strip()
            )
            continue
        if low == ":entities":
            _print_entities(last_ents)
            continue
        if low.startswith(":graph"):
            parts = user.split()
            if len(parts) < 2:
                print("Usage: :graph <entity_id|index>")
                continue
            target = parts[1]
            entity_id: Optional[str] = None
            if target.isdigit():
                idx = int(target) - 1
                if 0 <= idx < len(last_ents):
                    entity_id = last_ents[idx][1]
            else:
                for _, eid, _ in last_ents:
                    if eid == target:
                        entity_id = eid
                        break
            if not entity_id:
                print("Entity not found in last list.")
                continue
            path = bot.visualize_graph(entity_id)
            print(f"Graph image saved at: {path}" if path else f"Unable to render graph for {entity_id}")
            continue

        # Normal generation (compare mode)
        baseline, augmented, ents = bot.respond_compare(user)
        last_ents = ents
        # Side-by-side style (stacked, with clear separators)
        print_side_by_side(baseline, augmented, delimiter="|", col_padding=5)

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="KG-LFM Inference — CLI (Compare Mode, Tools-Only)")
    p.add_argument("--model_path", type=str, required=True, help="Path to the trained KG-LFM model")
    p.add_argument("--config", type=str, default=None, help="Path to dataset configuration YAML file")
    p.add_argument("--lite", action="store_true", help="Use lite dataset")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a helpful assistant answering the user queries."
        ),
        help="System prompt to seed the conversation",
    )

    args = p.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.loglevel))

    # Dataset config
    if args.config:
        from KG_LM.configuration import load_yaml_config

        cfg = load_yaml_config(args.config)
        dataset_cfg: DatasetConfig = cfg.dataset
        if args.lite:
            dataset_cfg.lite = True
    else:
        dataset_cfg = DatasetConfig(lite=args.lite)

    try:
        bot = KGChatBot(
            model_path=args.model_path,
            dataset_config=dataset_cfg,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            system_prompt=args.system_prompt,
        )
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    try:
        run_cli(bot)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
    except Exception as e:  # pragma: no cover
        logger.error(f"Error occurred while running CLI: {e}")
        raise

if __name__ == "__main__":
    main()
