"""
KG Decoder for Knowledge Graph Language Foundation Model (KG_LM).

The decoder performs two main tasks:
1) Relation prediction: Given a center node and surrounding nodes, predict the relations.
2) Object prediction: Given a center node and a relation, score candidate objects in the SAME graph.

Fixes:
- Candidate-aware object scoring.
- Correct feature dimensionality for the object predictor MLP.
- Relation projected into node space via a learned projection.
- Candidates selected per-graph (using Batch.batch).
- Self-supervised targets tie (center, relation) to true neighbor objects within each graph.
- NEW: Pad object_candidates to max_objects and carry an object_candidate_mask; masked
       candidates get score -inf so BCE ignores them.
"""

from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


class KGDecoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        final_embedding_dim: int,
        dropout: float = 0.2,
        num_quantizers: int = 3,
        edge_prediction_weight: float = 1.0,
        object_prediction_weight: float = 1.0,
        max_objects: int = 50,
        use_dot_score: bool = False,
    ):
        super().__init__()

        self.max_objects = max_objects

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.num_quantizers = num_quantizers
        self.edge_prediction_weight = edge_prediction_weight
        self.object_prediction_weight = object_prediction_weight
        self.use_dot_score = use_dot_score

        # Project quantized tokens back to node embedding space
        self.quantized_projection = nn.Sequential(
            nn.Linear(final_embedding_dim, 2 * node_embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
            nn.LayerNorm(node_embedding_dim),
        )

        # Aggregate multiple quantizers
        self.quantizer_aggregation = nn.Sequential(
            nn.Linear(num_quantizers * node_embedding_dim, node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(node_embedding_dim),
        )

        # Edge/Relation prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(3 * node_embedding_dim, 2 * node_embedding_dim),  # [ni, nj, ni - nj]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_embedding_dim, edge_embedding_dim),
        )

        # Relation projection to node space (edge_dim -> node_dim)
        self.rel_proj = nn.Linear(edge_embedding_dim, node_embedding_dim, bias=False)

        # Object prediction head (candidate-aware)
        if not self.use_dot_score:
            # Features: [q, candidate, q*candidate] -> 3 * node_dim
            self.object_predictor = nn.Sequential(
                nn.Linear(3 * node_embedding_dim, 2 * node_embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * node_embedding_dim, 1),
            )
        else:
            self.object_predictor = None

        self.dropout = nn.Dropout(dropout)

    # --------------------------
    # Utilities
    # --------------------------

    @staticmethod
    def _nodes_of_graph(batch_vec: torch.Tensor, gidx: int) -> torch.Tensor:
        return (batch_vec == gidx).nonzero(as_tuple=True)[0]

    # --------------------------
    # Self-supervised targets
    # --------------------------

    def create_self_supervised_targets(
        self,
        original_graphs: Optional[Batch],
        center_embeddings: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Build node_pairs/edge_targets and per-(center, relation) object candidates/targets
        from the *same* graph. All rows are padded/truncated to K=self.max_objects.

        Returns:
          dict with:
            'node_pairs': LongTensor [N, 2]
            'edge_targets': Tensor [N, edge_dim]
            'relations': Tensor [M, edge_dim]
            'relation_graph_indices': LongTensor [M]
            'object_candidates': Tensor [M, K, node_dim]
            'object_candidate_mask': BoolTensor [M, K]  (True = real candidate, False = pad)
            'object_targets': Tensor [M, K] (binary, float)
        """
        targets: Dict[str, Any] = {}
        if original_graphs is None:
            return targets

        device = center_embeddings.device

        has_edges = hasattr(original_graphs, "edge_index") and original_graphs.edge_index is not None
        has_edge_attr = hasattr(original_graphs, "edge_attr") and original_graphs.edge_attr is not None
        has_x = hasattr(original_graphs, "x") and original_graphs.x is not None
        has_batch = hasattr(original_graphs, "batch") and original_graphs.batch is not None
        if not (has_edges and has_edge_attr and has_x and has_batch):
            return targets

        edge_index = original_graphs.edge_index.to(device)  # [2, E]
        edge_attr = original_graphs.edge_attr.to(device)    # [E, edge_dim]
        X = original_graphs.x.to(device)                    # [N, node_dim]
        batch_vec = original_graphs.batch.to(device)        # [N]

        B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0

        node_pairs: List[torch.Tensor] = []
        edge_targets: List[torch.Tensor] = []
        relations: List[torch.Tensor] = []
        relation_graph_indices: List[int] = []
        object_candidates_rows: List[torch.Tensor] = []
        object_candidate_mask_rows: List[torch.Tensor] = []
        object_targets_rows: List[torch.Tensor] = []

        for g in range(B):
            nodes_g = self._nodes_of_graph(batch_vec, g)
            if nodes_g.numel() == 0:
                continue

            center_global = int(nodes_g[0].item())  # first node as center

            # Prefer outgoing edges center -> neighbor, else incoming
            mask_out = edge_index[0] == center_global
            e_out = mask_out.nonzero(as_tuple=True)[0]
            use_incoming = False
            if e_out.numel() == 0:
                mask_in = edge_index[1] == center_global
                e_in = mask_in.nonzero(as_tuple=True)[0]
                if e_in.numel() == 0:
                    continue
                use_incoming = True

            # Build fixed-size candidate bank for this graph (padded to max_objects)
            g_node_count = int(nodes_g.numel())
            K_real = min(self.max_objects, g_node_count)
            cand_idx_real = nodes_g[:K_real]                    # [K_real]
            cand_emb_real = X[cand_idx_real]                    # [K_real, node_dim]

            if K_real < self.max_objects:
                # pad candidates with zeros
                pad_c = torch.zeros(self.max_objects - K_real, X.shape[1], device=device, dtype=X.dtype)
                cand_emb = torch.cat([cand_emb_real, pad_c], dim=0)  # [K, node_dim]
                cand_mask = torch.cat(
                    [torch.ones(K_real, device=device, dtype=torch.bool),
                     torch.zeros(self.max_objects - K_real, device=device, dtype=torch.bool)],
                    dim=0
                )  # [K]
                cand_idx_padded = torch.cat(
                    [cand_idx_real,
                     torch.full((self.max_objects - K_real,), -1, device=device, dtype=torch.long)],
                    dim=0
                )  # [K] (indices -1 for pads, only used to place positives)
            else:
                cand_emb = cand_emb_real                         # [K, node_dim]
                cand_mask = torch.ones(self.max_objects, device=device, dtype=torch.bool)  # all real
                cand_idx_padded = cand_idx_real                  # [K]

            # Iterate edges tied to center
            edges_for_center = e_out if not use_incoming else e_in
            for e in edges_for_center.tolist():
                neighbor_global = int(edge_index[1 - int(use_incoming), e].item())

                # For edge prediction
                node_pairs.append(torch.tensor([center_global, neighbor_global], device=device, dtype=torch.long))
                edge_targets.append(edge_attr[e])

                # Relation rows
                relations.append(edge_attr[e])
                relation_graph_indices.append(g)

                # Object targets over padded candidate set
                obj_tgt = torch.zeros(self.max_objects, device=device)
                # mark positive if neighbor appears in the real part of the candidate list
                where = (cand_idx_padded == neighbor_global).nonzero(as_tuple=True)[0]
                if where.numel() > 0:
                    obj_tgt[int(where[0].item())] = 1.0

                object_candidates_rows.append(cand_emb)         # [K, node_dim] K=max_objects
                object_candidate_mask_rows.append(cand_mask)    # [K]
                object_targets_rows.append(obj_tgt)             # [K]

        # Stack
        if node_pairs:
            targets["node_pairs"] = torch.stack(node_pairs, dim=0)  # [N, 2]
        if edge_targets:
            targets["edge_targets"] = torch.stack(edge_targets, dim=0)  # [N, edge_dim]
        if relations:
            targets["relations"] = torch.stack(relations, dim=0)  # [M, edge_dim]
            targets["relation_graph_indices"] = torch.tensor(relation_graph_indices, device=device, dtype=torch.long)
        if object_candidates_rows:
            targets["object_candidates"] = torch.stack(object_candidates_rows, dim=0)      # [M, K, node_dim]
            targets["object_candidate_mask"] = torch.stack(object_candidate_mask_rows, 0)  # [M, K] (bool)
        if object_targets_rows:
            targets["object_targets"] = torch.stack(object_targets_rows, dim=0)            # [M, K]

        return targets

    # --------------------------
    # Quantized token aggregation
    # --------------------------

    def aggregate_quantized_tokens(self, quantized_tokens: torch.Tensor) -> torch.Tensor:
        B, Q, D = quantized_tokens.shape
        assert Q == self.num_quantizers, f"Expected {self.num_quantizers} quantizers, got {Q}"

        projected = []
        for i in range(Q):
            # Ensure quantized tokens have the same dtype as the projection parameters
            target_dtype = next(self.quantized_projection.parameters()).dtype
            tokens_i = quantized_tokens[:, i, :].to(dtype=target_dtype)
            proj_i = self.quantized_projection(tokens_i)  # [B, node_dim]
            projected.append(proj_i)
        concatenated = torch.cat(projected, dim=1)                         # [B, Q*node_dim]
        # Ensure concatenated tensor has the same dtype as the aggregation parameters
        target_dtype = next(self.quantizer_aggregation.parameters()).dtype
        concatenated = concatenated.to(dtype=target_dtype)
        aggregated = self.quantizer_aggregation(concatenated)              # [B, node_dim]
        return aggregated

    # --------------------------
    # Edge prediction
    # --------------------------

    def predict_edges(
        self,
        center_embeddings: torch.Tensor,
        node_pairs: Optional[torch.Tensor],
        original_graphs: Optional[Batch] = None,
    ) -> torch.Tensor:
        device = center_embeddings.device
        if node_pairs is None or (isinstance(node_pairs, torch.Tensor) and node_pairs.numel() == 0):
            if original_graphs is None or not hasattr(original_graphs, "edge_index"):
                return torch.empty(0, self.edge_embedding_dim, device=device)
            edge_index = original_graphs.edge_index
            if edge_index is None or edge_index.numel() == 0:
                return torch.empty(0, self.edge_embedding_dim, device=device)
            num_pairs = min(edge_index.shape[1], center_embeddings.shape[0] * 2)
            node_pairs = edge_index[:, :num_pairs].t().contiguous()

        graph_node_embeddings = None
        if original_graphs is not None and hasattr(original_graphs, "x") and original_graphs.x is not None:
            graph_node_embeddings = original_graphs.x.to(device)

        B = center_embeddings.shape[0]
        feats = []
        for idx, (i, j) in enumerate(node_pairs.tolist()):
            if graph_node_embeddings is not None and i < graph_node_embeddings.shape[0] and j < graph_node_embeddings.shape[0]:
                ni = graph_node_embeddings[i]
                nj = graph_node_embeddings[j]
            else:
                ni = center_embeddings[idx % B]
                nj = center_embeddings[(idx + 1) % B]
            feats.append(torch.cat([ni, nj, ni - nj], dim=-1))

        if not feats:
            return torch.empty(0, self.edge_embedding_dim, device=device)

        feats = torch.stack(feats, dim=0)                              # [N, 3*node_dim]
        # Ensure feats has the same dtype as the edge_predictor parameters
        target_dtype = next(self.edge_predictor.parameters()).dtype
        original_dtype = feats.dtype
        feats = feats.to(dtype=target_dtype)
        edge_predictions = self.edge_predictor(feats)                  # [N, edge_dim]
        return edge_predictions.to(dtype=original_dtype)

    # --------------------------
    # Object prediction
    # --------------------------

    def _score_candidates(
        self,
        query_vec: torch.Tensor,       # [node_dim]
        candidates: torch.Tensor,      # [K, node_dim]
    ) -> torch.Tensor:
        if self.use_dot_score:
            return candidates @ query_vec  # [K]
        else:
            K = candidates.shape[0]
            q_exp = query_vec.unsqueeze(0).expand(K, -1)
            feats = torch.cat([q_exp, candidates, q_exp * candidates], dim=-1)  # [K, 3*node_dim]
            # Ensure feats has the same dtype as the object_predictor parameters
            target_dtype = next(self.object_predictor.parameters()).dtype
            original_dtype = feats.dtype
            feats = feats.to(dtype=target_dtype)
            return self.object_predictor(feats).squeeze(-1).to(dtype=original_dtype)  # [K]

    def predict_objects(
        self,
        center_embeddings: torch.Tensor,                 # [B, node_dim]
        relations: Optional[torch.Tensor],               # [M, edge_dim]
        object_candidates: Optional[torch.Tensor] = None,# [M, K, node_dim]
        original_graphs: Optional[Batch] = None,
        relation_graph_indices: Optional[torch.Tensor] = None,  # [M]
        object_candidate_mask: Optional[torch.Tensor] = None,   # [M, K] (bool)
    ) -> torch.Tensor:
        device = center_embeddings.device

        if relations is None or (isinstance(relations, torch.Tensor) and relations.numel() == 0):
            return torch.empty(0, self.max_objects, device=device)

        M = relations.shape[0]
        B = center_embeddings.shape[0]

        predictions: List[torch.Tensor] = []

        if object_candidates is None:
            # Build candidates per relation from original_graphs (variable K'); we will pad to K.
            if original_graphs is None or not hasattr(original_graphs, "x") or not hasattr(original_graphs, "batch"):
                # Fallback: use noisy centers
                base = center_embeddings
                K = min(self.max_objects, base.shape[0])
                base = base[:K]
                if K < self.max_objects:
                    reps = (self.max_objects + K - 1) // K
                    base = base.repeat(reps, 1)[: self.max_objects]
                for r in range(M):
                    center_vec = center_embeddings[r % B]
                    # Ensure relations tensor has the same dtype as rel_proj parameters
                    target_dtype = next(self.rel_proj.parameters()).dtype
                    original_dtype = relations.dtype
                    relation_vec = relations[r].to(dtype=target_dtype)
                    q = center_vec + self.rel_proj(relation_vec)
                    scores = self._score_candidates(q, base)
                    scores = scores.to(dtype=original_dtype)
                    if scores.shape[0] < self.max_objects:
                        pad = torch.full((self.max_objects - scores.shape[0],), -float("inf"), device=device)
                        scores = torch.cat([scores, pad], dim=0)
                    predictions.append(scores)
            else:
                X = original_graphs.x.to(device)
                batch_vec = original_graphs.batch.to(device)
                for r in range(M):
                    gidx = int(relation_graph_indices[r].item()) if relation_graph_indices is not None else (r % B)
                    nodes_g = self._nodes_of_graph(batch_vec, gidx)
                    if nodes_g.numel() == 0:
                        # fallback to center bank
                        base = center_embeddings
                        K = min(self.max_objects, base.shape[0])
                        cands = base[:K]
                    else:
                        K = min(self.max_objects, int(nodes_g.numel()))
                        cands = X[nodes_g[:K]]
                    center_vec = center_embeddings[gidx]
                    # Ensure relations tensor has the same dtype as rel_proj parameters
                    target_dtype = next(self.rel_proj.parameters()).dtype
                    original_dtype = relations.dtype
                    relation_vec = relations[r].to(dtype=target_dtype)
                    q = center_vec + self.rel_proj(relation_vec)
                    scores = self._score_candidates(q, cands).to(dtype=original_dtype)  # [K]
                    if scores.shape[0] < self.max_objects:
                        pad = torch.full((self.max_objects - scores.shape[0],), -float("inf"), device=device)
                        scores = torch.cat([scores, pad], dim=0)
                    elif scores.shape[0] > self.max_objects:
                        scores = scores[: self.max_objects]
                    predictions.append(scores)
        else:
            # Provided fixed-size [M, K=self.max_objects, node_dim]; use mask to -inf the pads
            for r in range(M):
                gidx = int(relation_graph_indices[r].item()) if relation_graph_indices is not None else (r % B)
                center_vec = center_embeddings[gidx]
                # Ensure relations tensor has the same dtype as rel_proj parameters
                target_dtype = next(self.rel_proj.parameters()).dtype
                original_dtype = relations.dtype
                relation_vec = relations[r].to(dtype=target_dtype)
                q = center_vec + self.rel_proj(relation_vec)
                cands = object_candidates[r]                            # [K, node_dim] with K=self.max_objects
                scores = self._score_candidates(q, cands).to(dtype=original_dtype)               # [K]
                if object_candidate_mask is not None:
                    mask_r = object_candidate_mask[r]                   # [K] bool
                    # set padded (False) positions to -inf
                    scores = scores.masked_fill(~mask_r, -float("inf"))
                # Ensure exactly K=self.max_objects (already true)
                predictions.append(scores)

        if predictions:
            return torch.stack(predictions, dim=0)  # [M, K]
        return torch.empty(0, self.max_objects, device=device)

    # --------------------------
    # Losses
    # --------------------------

    @staticmethod
    def compute_edge_prediction_loss(edge_predictions: torch.Tensor, edge_targets: torch.Tensor) -> torch.Tensor:
        if edge_predictions.numel() == 0 or edge_targets.numel() == 0:
            return edge_predictions.new_tensor(0.0)
        return F.mse_loss(edge_predictions, edge_targets)

    @staticmethod
    def compute_object_prediction_loss(object_predictions: torch.Tensor, object_targets: torch.Tensor) -> torch.Tensor:
        if object_predictions.numel() == 0 or object_targets.numel() == 0:
            return object_predictions.new_tensor(0.0)
        # Mask padded positions (=-inf)
        valid_mask = ~torch.isinf(object_predictions)
        if valid_mask.sum() == 0:
            return object_predictions.new_tensor(0.0)
        valid_preds = object_predictions[valid_mask]
        valid_tgts = object_targets[valid_mask].float()
        return F.binary_cross_entropy_with_logits(valid_preds, valid_tgts, reduction="mean")

    # --------------------------
    # Forward
    # --------------------------

    def forward(
        self,
        quantized_tokens: torch.Tensor,                 # [B, Q, D]
        original_graphs: Optional[Batch] = None,
        node_pairs: Optional[torch.Tensor] = None,      # [N, 2]
        relations: Optional[torch.Tensor] = None,       # [M, edge_dim]
        object_candidates: Optional[torch.Tensor] = None,      # [M, K, node_dim]
        edge_targets: Optional[torch.Tensor] = None,    # [N, edge_dim]
        object_targets: Optional[torch.Tensor] = None,  # [M, K]
        mode: str = "both",
    ) -> Dict[str, Any]:
        device = quantized_tokens.device
        results: Dict[str, Any] = {}

        # Aggregate quantized tokens into center node representations
        center_embeddings = self.aggregate_quantized_tokens(quantized_tokens)  # [B, node_dim]
        results["center_embeddings"] = center_embeddings

        total_loss = torch.tensor(0.0, device=device)

        # Build self-supervised targets if missing
        relation_graph_indices = None
        object_candidate_mask = None
        if original_graphs is not None:
            sst = self.create_self_supervised_targets(original_graphs, center_embeddings)
            node_pairs = sst.get("node_pairs", node_pairs)
            relations = sst.get("relations", relations)
            edge_targets = sst.get("edge_targets", edge_targets)
            object_targets = sst.get("object_targets", object_targets)
            object_candidates = sst.get("object_candidates", object_candidates)
            relation_graph_indices = sst.get("relation_graph_indices", relation_graph_indices)
            object_candidate_mask = sst.get("object_candidate_mask", object_candidate_mask)

        # Edge / Relation prediction
        if mode in ("edge_prediction", "both"):
            edge_predictions = self.predict_edges(center_embeddings, node_pairs, original_graphs)
            results["edge_predictions"] = edge_predictions

            if edge_targets is not None and edge_predictions.numel() > 0:
                # Align counts
                if edge_targets.shape[0] > edge_predictions.shape[0]:
                    edge_targets = edge_targets[: edge_predictions.shape[0]]
                elif edge_targets.shape[0] < edge_predictions.shape[0]:
                    reps = (edge_predictions.shape[0] + edge_targets.shape[0] - 1) // edge_targets.shape[0]
                    edge_targets = edge_targets.repeat(reps, 1)[: edge_predictions.shape[0]]

                e_loss = self.compute_edge_prediction_loss(edge_predictions, edge_targets)
                results["edge_loss"] = e_loss
                total_loss = total_loss + self.edge_prediction_weight * e_loss

        # Object prediction
        if mode in ("object_prediction", "both"):
            obj_preds = self.predict_objects(
                center_embeddings=center_embeddings,
                relations=relations,
                object_candidates=object_candidates,
                original_graphs=original_graphs,
                relation_graph_indices=relation_graph_indices,
                object_candidate_mask=object_candidate_mask,
            )
            results["object_predictions"] = obj_preds

            if object_targets is not None and obj_preds.numel() > 0:
                # Ensure targets match [M, K]
                M_pred, K_pred = obj_preds.shape
                if object_targets.shape[0] > M_pred:
                    object_targets = object_targets[:M_pred]
                elif object_targets.shape[0] < M_pred:
                    pad_rows = torch.zeros(
                        (M_pred - object_targets.shape[0], object_targets.shape[1]),
                        device=object_targets.device,
                    )
                    object_targets = torch.cat([object_targets, pad_rows], dim=0)

                if object_targets.shape[1] != K_pred:
                    if object_targets.shape[1] < K_pred:
                        pad_cols = torch.zeros(
                            (object_targets.shape[0], K_pred - object_targets.shape[1]),
                            device=object_targets.device,
                        )
                        object_targets = torch.cat([object_targets, pad_cols], dim=1)
                    else:
                        object_targets = object_targets[:, :K_pred]

                o_loss = self.compute_object_prediction_loss(obj_preds, object_targets)
                results["object_loss"] = o_loss
                total_loss = total_loss + self.object_prediction_weight * o_loss

        if total_loss.item() > 0:
            results["total_loss"] = total_loss

        return results
