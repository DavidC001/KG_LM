"""
pytorch model that when received a Batch of graphs,
returns for each the residual vector quantized representsion for each graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import logging

# graph pooling
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import Batch
from vector_quantize_pytorch import ResidualVQ

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
    codebook_size : int, default 512
        Number of entries in each codebook.
    shared_codebook : bool, default False
        If True, every stage re-uses the same nn.Embedding object.
    commit_weight : float, default 0.25
        Multiplier for the commitment (codebook) loss term.
    """
    def __init__(
        self,
        dim,
        num_quantizers: int = 3,
        codebook_size: int = 512,
        shared_codebook: bool = False,
        commit_weight: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.commit_weight = commit_weight

        # ── codebooks ──────────────────────────────────────────────────────────
        if shared_codebook:
            shared = nn.Embedding(codebook_size, dim)
            self.codebooks = nn.ModuleList([shared] * num_quantizers)
        else:
            self.codebooks = nn.ModuleList(
                [nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)]
            )
            
    def _init_weights(self):
        """Initialize weights with proper scaling for stability."""
        for cb in self.codebooks:
            nn.init.xavier_uniform_(cb.weight, gain=0.1)

    # -------- helpers ---------------------------------------------------------
    @torch.no_grad()
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
        dist = torch.cdist(residual.unsqueeze(1), table.weight.unsqueeze(0)).squeeze(1)
        idx = dist.argmin(-1)           # (B,)
        return idx, table(idx)          # (B,), (B, D)

    # -------- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : (B, D) – encoder embeddings
        
        Returns
        -------
        quantized : (B, Q, D)   stacked quantized vectors with STE for gradient flow
        indices   : (B, Q)      codebook indices per stage
        loss      : ()          scalar — residual-energy + β * commit loss
        """
        assert x.dim() == 2 and x.size(1) == self.dim, \
            f"expected (batch, {self.dim}); got {tuple(x.shape)}"

        residual = x
        quantized_list, index_list = [], []
        commit_loss = 0.0

        for cb in self.codebooks:
            # 1. Pick the nearest codebook vector (this operation is non-differentiable)
            idx, q = self._nearest_code(residual, cb)
            
            # 2. Apply the Straight-Through Estimator (STE)
            # This allows the gradient to be passed from 'q_ste' back to 'residual'
            # as if the operation was an identity function in the backward pass.
            q_ste = residual + (q - residual).detach()
            
            # 3. Accumulate the STE-modified quantized vectors for the final output
            quantized_list.append(q_ste)
            index_list.append(idx)
            
            # 4. Calculate the commitment loss to train the codebook.
            # The gradient for this loss should only affect the codebook, not the encoder.
            commit_loss = commit_loss + F.mse_loss(residual.detach(), q)
            
            # 5. Update the residual for the next quantization stage using the original 'q'.
            residual = residual - q

        quantized = torch.stack(quantized_list, dim=1)   # (B, Q, D)
        indices   = torch.stack(index_list,   dim=1)     # (B, Q)
        
        # The residual loss encourages the encoder to produce outputs that are 
        # close to the codebook, providing a gradient path back to the encoder.
        res_loss  = residual.pow(2).mean()

        # The total loss for the quantization module
        loss = res_loss + self.commit_weight * commit_loss

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
        
        # adapter layer
        self.adapter = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.dropout = nn.Dropout(dropout)

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
        
        # output projection layer
        self.output_projection = nn.Linear(node_embedding_dim, final_embedding_dim)
        
        self.graph_pooling = graph_pooling
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling for stability."""
        # Initialize adapter layer with small weights
        nn.init.xavier_uniform_(self.adapter.weight, gain=0.1)
        nn.init.constant_(self.adapter.bias, 0)
        
        # Initialize output projection with small weights
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.constant_(self.output_projection.bias, 0)
        
        # Initialize layer norm
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)
        
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
        
        # Apply GATv2Conv
        x = self.conv(x, edge_index, edge_attr)
        
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
        
        logging.debug(f"Output shape after adapter and normalization: {x.shape}")
        # Apply residual vector quantization
        quantized_x, indices, loss = self.vq(x)
        logging.debug(f"Quantized output shape: {quantized_x.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
        
        tokens = self.output_projection(quantized_x)
        
        tokens = self.dropout(tokens)
        
        # Normalize the quantized output
        tokens = self.out_norm(tokens)
        
        # Return the quantized representations
        logging.debug(f"Quantized output shape: {tokens.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
        return tokens, indices, loss