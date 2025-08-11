"""
pytorch model that when received a Batch of graphs,
returns for each the residual vector quantized representsion for each graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn.utils import spectral_norm
import logging

# graph pooling
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import Batch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantization(nn.Module):
    """
    Residual-vector quantizer that outputs one code per stage.
    
    Parameters
    ----------
    dim : int
        Embedding dimension of the inputs.
    num_quantizers : int, default 3
        Number of residual stages (codebooks).
    codebook_size : int, default 256
        Number of entries in each codebook.
    shared_codebook : bool, default False
        If True, every stage re-uses the same nn.Embedding object.
    scale_commit: float, default 0.25
        Scaling factor for the commitment loss on the overall loss.
    beta_commit : float, default 0.25
        Weight for the commitment loss on the encoder side.
    gamma_entropy : float, default 1.0
        Weight for the entropy regularization loss.
    lambda_res : float, default 0.5
        Weight for the residual loss.
    temperature : float, default 1.0
    """
    def __init__(
        self,
        dim,
        num_quantizers: int = 3,
        codebook_size: int = 256,
        shared_codebook: bool = False,
        scale_commit=0.25,
        beta_commit=0.25, gamma_entropy=1.0, lambda_res=0.5,
        temperature: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        self.scale_commit = scale_commit
        self.beta_commit = beta_commit
        self.gamma_entropy = gamma_entropy
        self.lambda_res = lambda_res
        self.temperature = temperature

        self.shared_codebook = shared_codebook

        # ── codebooks ──────────────────────────────────────────────────────────
        if shared_codebook:
            self.shared = nn.Embedding(codebook_size, dim)
            self.codebooks = nn.ModuleList([self.shared])  # length 1
        else:
            self.codebooks = nn.ModuleList(
                [nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)]
            )
            
        self._init_weights()
            
    def _init_weights(self):
        """Initialize weights with proper scaling for stability."""
        for cb in self.codebooks:
            nn.init.xavier_uniform_(cb.weight, gain=0.1)

    # -------- helpers ---------------------------------------------------------
    def _nearest_code(self, residual: torch.Tensor, table: nn.Embedding):
        """
        Args
        ----
        residual : (B, D)
        table.weight : (K, D)
        
        Returns
        -------
        idx : (B,)          LongTensor – chosen code indices
        q   : (B, D)        quantized vectors
        """
        # (B, 1, D) × (1, K, D) → (B, K)
        x = residual.float()                    # (B,D)
        w = table.weight.float()                # (K,D)
        x2 = (x * x).sum(-1, keepdim=True)      # (B,1)
        w2 = (w * w).sum(-1).unsqueeze(0)       # (1,K)
        dist2 = x2 + w2 - 2 * (x @ w.t())       # (B,K)

        idx = dist2.argmin(-1)           # (B,)
        return idx, table(idx), dist2    # (B,), (B, D), (B, K)

    @torch.no_grad()
    def _compute_adaptive_entropy_weight(self, commit_loss, perplexity, batch_diversity):
        """
        Compute adaptive entropy weight based on commitment loss, perplexity, and batch diversity.
        
        Args:
            commit_loss: Current commitment loss
            perplexity: Current perplexity of the codebook usage
            batch_diversity: Measure of diversity in the current batch
        
        Returns:
            Adaptive entropy weight
        """
        # Normalize metrics to [0, 1] range for stable scaling
        commit_norm = torch.sigmoid(commit_loss * 10)  # Scale commitment loss
        perplexity_norm = perplexity / self.codebook_size  # Normalize by codebook size
        diversity_norm = torch.clamp(batch_diversity, 0, 1)  # Ensure in [0, 1]
        
        # Scale up entropy when:
        # - Low batch diversity (need more exploration)
        # - Low perplexity (codebook underutilized)
        # Scale down entropy when:
        # - High commitment loss and high perplexity (codebook well utilized)
        
        # Base entropy weight
        base_weight = self.gamma_entropy
        
        # Diversity factor: increase entropy when diversity is low
        diversity_factor = 2.0 - diversity_norm  # Range [1, 2]
        
        # Commitment-perplexity factor: decrease entropy when both are high
        commit_perp_factor = 1.0 / (1.0 + commit_norm * perplexity_norm)  # Range (0, 1]
        
        # Combine factors
        adaptive_weight = base_weight * diversity_factor * commit_perp_factor
        
        return adaptive_weight

    def _compute_batch_diversity(self, x):
        """
        Compute diversity measure for the batch.
        Uses pairwise distances between samples.
        """
        if x.size(0) <= 1:
            return torch.tensor(0.0, device=x.device)
        
        # torch.cdist has incomplete support for bfloat16/float16 on some builds (raises: "cdist_cuda" not implemented for 'BFloat16').
        # Temporarily upcast to float32 for the distance computation, then continue.
        needs_cast = x.dtype in (torch.bfloat16, torch.float16)
        x_for_dist = x.float() if needs_cast else x
        
        pairwise_dist = torch.cdist(x_for_dist, x_for_dist)
        
        # Remove diagonal (self-distances)
        mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        distances = pairwise_dist[mask]
        
        # Diversity as normalized mean distance (use same precision tensor as distances)
        denom = (x_for_dist.norm(dim=1).mean() + 1e-8)
        diversity = distances.mean() / denom
        
        return torch.clamp(diversity.to(x.device), 0, 1)

    # -------- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : (B, D) – encoder embeddings
        
        Returns
        -------
        quantized : (B, Q, D)   stacked quantized vectors with STE for gradient flow
        indices   : (B, Q)      codebook indices per stage
        loss      : ()          scalar — residual + β * commit loss - entropy
        """
        assert x.dim() == 2 and x.size(1) == self.dim, \
            f"expected (batch, {self.dim}); got {tuple(x.shape)}"

        residual = x
        quantized_list, index_list = [], []
        
        if self.training:
            total_commit = 0.0
            total_entropy = 0.0

            # Compute batch diversity once
            with torch.no_grad():
                batch_diversity = self._compute_batch_diversity(x)

        for i in range(self.num_quantizers):
            cb = self.codebooks[0] if self.shared_codebook else self.codebooks[i]
            
            # 1. Pick the nearest codebook vector (this operation is non-differentiable)
            idx, q, dist2 = self._nearest_code(residual, cb)

            # 2. Apply the Straight-Through Estimator (STE)
            # This allows the gradient to be passed from 'q_ste' back to 'residual'
            # as if the operation was an identity function in the backward pass.
            q_ste = residual + (q - residual).detach()
            
            # 3. Accumulate the STE-modified quantized vectors for the final output
            quantized_list.append(q_ste)
            index_list.append(idx)
            
            if self.training:
                # 4. Calculate the commitment loss.
                commit_loss = F.mse_loss(residual.detach(), q) + self.beta_commit * F.mse_loss(residual, q.detach())

                # 5. Entropy regularization for better codebook utilization
                prob_dist = F.softmax(-dist2 / self.temperature, dim=-1)
                entropy_loss = -(prob_dist * torch.log(prob_dist + 1e-8)).sum(-1).mean()
                
                # 6. Compute perplexity for adaptive scaling
                # Perplexity measures how well the probability distribution is spread
                perplexity = torch.exp(entropy_loss)
                
                # 7. Compute adaptive entropy weight
                adaptive_entropy_weight = self._compute_adaptive_entropy_weight(
                    commit_loss, perplexity, batch_diversity
                )
            
                # 8. Accumulate the total loss with adaptive entropy weight
                total_commit = total_commit + commit_loss
                # need to use - as we want to maximize the entropy
                total_entropy = total_entropy - adaptive_entropy_weight * entropy_loss

            # 9. Update the residual for the next quantization stage using the original 'q'.
            residual = residual - q
            
        quantized = torch.stack(quantized_list, dim=1)   # (B, Q, D)
        indices   = torch.stack(index_list,   dim=1)     # (B, Q)
        
        if self.training:
            total_commit = self.scale_commit * total_commit / self.num_quantizers
            total_entropy = total_entropy / self.num_quantizers
            
            # The residual loss encourages the encoder to produce outputs that are 
            # close to the codebook.
            res_loss = residual.pow(2).mean() * self.lambda_res

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Residual loss: %s, commit loss: %s, entropy loss: %s, batch diversity: %s",
                            res_loss.item(), total_commit.item(), total_entropy.item(), batch_diversity.item())
            # The total loss for the quantization module
            loss = res_loss + total_commit + total_entropy
        else:
            # During inference, we do not compute the loss (we are interested on the language loss, not the quantization loss)
            loss = torch.tensor(0.0, device=x.device)

        return quantized, indices, loss


class KGEncoder(nn.Module):
    def __init__(
        self, 
        node_embedding_dim, 
        edge_embedding_dim,
        final_embedding_dim,
        dropout=0.2,
        num_heads=1,
        num_quantizers=3,
        codebook_size=512,
        shared_codebook=False,
        graph_pooling=True
    ):
        """
        Initializes the KGEncoder model.
        
        Args:
            node_embedding_dim (int): Dimension of the node embeddings.
            edge_embedding_dim (int): Dimension of the edge embeddings.
            final_embedding_dim (int): Dimension of the final output embeddings.
            dropout (float): Dropout rate for the model.
            num_heads (int): Number of attention heads in GATv2Conv.
            num_quantizers (int): Number of quantizers for residual vector quantization.
            codebook_size (int): Size of the codebook for vector quantization.
            shared_codebook (bool): Whether to use a shared codebook across quantizers.
            graph_pooling (bool): Whether to apply global mean pooling to the graph representations.
            
        """
        super(KGEncoder, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim

        self.conv = GATv2Conv(
            in_channels=node_embedding_dim,
            out_channels=node_embedding_dim,
            edge_dim=edge_embedding_dim,
            heads=num_heads,
            dropout=0.2,
        )
        self.num_heads = num_heads
        
        # adapter layer with spectral normalization for stability
        self.adapter = spectral_norm(nn.Linear(node_embedding_dim, node_embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = nn.Dropout(dropout * 0.5)  # Edge-specific dropout
        self.attention_dropout = nn.Dropout(dropout * 0.3)  # Attention dropout

        # normalization layer
        self.norm = nn.LayerNorm(node_embedding_dim)
        self.out_norm = nn.LayerNorm(final_embedding_dim)

        # residual vector quantization layer
        self.vq = ResidualVectorQuantization(
            dim=node_embedding_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            shared_codebook=shared_codebook
        )
        
        # output projection layer with spectral normalization
        self.output_projection = spectral_norm(nn.Linear(node_embedding_dim, final_embedding_dim))
        
        self.graph_pooling = graph_pooling
        
    def forward(self, graphs: Batch):
        """
        Forward pass for the KGEncoder.
        
        Args:
            graphs (Batch): A batch of graphs from PyTorch Geometric.
        
        Returns:
            torch.Tensor: The quantized representations of the graphs.
        """
        logging.debug("Forward into KG_encoder")
        x, edge_index, edge_attr = graphs.x, graphs.edge_index, graphs.edge_attr
        
        logging.debug("Got graph data")
        
        x = x.to(dtype=self.conv.lin_l.weight.dtype)
        edge_attr = edge_attr.to(dtype=self.conv.lin_l.weight.dtype)
        
        #TODO: check if this works in a distributed setting
        x = x.to(self.conv.lin_l.weight.device)
        edge_index = edge_index.to(self.conv.lin_l.weight.device)
        edge_attr = edge_attr.to(self.conv.lin_l.weight.device)
        
        logging.debug("Moved tensors")
        
        # Apply edge dropout before GATv2Conv
        if self.training:
            edge_attr = self.edge_dropout(edge_attr)
        
        # Apply GATv2Conv
        x = self.conv(x, edge_index, edge_attr)
        
        # Apply attention dropout after GAT
        if self.training:
            x = self.attention_dropout(x)
        
        logging.debug("Applied GATv2Conv")
        
        # average the output across all heads
        x = x.view(x.shape[0], self.num_heads, -1).mean(dim=1)
        
        logging.debug("reshaped tensors")
        
        if self.graph_pooling:
            x = global_mean_pool(x, graphs.batch.to(x.device))
        else:
            # Find the first occurrence of each batch index
            num_graphs = graphs.batch.max().item() + 1
            
            # Find first node index for each graph in the batch (GPU-optimized, no loops)
            first_indices = torch.zeros(num_graphs, dtype=torch.long, device=graphs.batch.device)
            mask = graphs.batch[:-1] != graphs.batch[1:] # Find where the batch index changes
            first_indices[1:] = torch.where(mask)[0] + 1 # Shift indices by 1 to account for the first node in each graph
            
            x = x[first_indices]
        
        logging.debug(f"Graph embeddings shape after GATv2Conv: {x.shape}") 
        
        # Apply adapter layer
        x = self.adapter(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Normalize the output
        x = self.norm(x)
        
        # Apply residual vector quantization
        quantized_x, indices, loss = self.vq(x)
        
        
        tokens = self.output_projection(quantized_x)
        
        tokens = self.dropout(tokens)
        
        # Normalize the quantized output
        tokens = self.out_norm(tokens)
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Output shape after adapter and normalization: {x.shape}")
            logging.debug(f"Quantized output shape: {quantized_x.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
            # Return the quantized representations
            logging.debug(f"Quantized output shape: {tokens.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
        return tokens, indices, loss