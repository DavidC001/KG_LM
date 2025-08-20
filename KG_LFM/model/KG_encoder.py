"""
pytorch model that when received a Batch of graphs,
returns for each the residual vector quantized representsion for each graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
import logging

from vector_quantize_pytorch import ResidualVQ

# graph pooling
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import Batch

import torch
import torch.nn as nn
import torch.nn.functional as F



class KGEncoder(nn.Module):
    def __init__(
        self, 
        node_embedding_dim, edge_embedding_dim, final_embedding_dim, std_tokens,
        dropout=0.2, num_heads=1, gat_layers=3, graph_pooling=True, 
        num_quantizers=3, codebook_size=512, codebook_dim=0, shared_codebook=False,
        beta_commit=0.25,
    ):
        """
        Initializes the KGEncoder model.
        
        Args:
            node_embedding_dim (int): Dimension of the node embeddings.
            edge_embedding_dim (int): Dimension of the edge embeddings.
            final_embedding_dim (int): Dimension of the final output embeddings.
            std_tokens (float): Standard deviation of the token embeddings.
            dropout (float): Dropout rate for the model.
            num_heads (int): Number of attention heads in GATv2Conv.
            gat_layers (int): Number of GATv2Conv layers.
            graph_pooling (bool): Whether to apply global mean pooling to the graph representations.
            num_quantizers (int): Number of quantizers for residual vector quantization.
            codebook_size (int): Size of the codebook for vector quantization.
            codebook_dim (int): Dimension of the downsampled codebook. If 0, no downsampling is applied.
            shared_codebook (bool): Whether to use a shared codebook across quantizers.
            beta_commit (float): Commitment loss weight for the vector quantization.
        """
        super(KGEncoder, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        
        self.convs = nn.ModuleList([
            TransformerConv(
                in_channels=node_embedding_dim,
                out_channels=node_embedding_dim // num_heads,
                heads=num_heads,
                edge_dim=edge_embedding_dim,
                dropout=dropout,
                concat=True
            )
            for _ in range(gat_layers)
        ])
        self.norms = nn.ModuleList([GraphNorm(node_embedding_dim) for _ in range(gat_layers)])

        self.num_heads = num_heads
        
        self.dropout = nn.Dropout(dropout)
        self.num_quantizers = num_quantizers

        if codebook_dim > 0:
            self.downsample = nn.Linear(node_embedding_dim, codebook_dim)
            self.downsample_norm = nn.LayerNorm(codebook_dim)
            self.upsample = nn.Linear(codebook_dim, node_embedding_dim)
            self.upsample_norm = nn.LayerNorm(node_embedding_dim)
        
        self.codebook_dim = codebook_dim

        self.vq = ResidualVQ(
            dim=node_embedding_dim if codebook_dim == 0 else codebook_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            shared_codebook=shared_codebook,
            use_cosine_sim = True,
            threshold_ema_dead_code = 2,
        )
        self.beta_commit = beta_commit
        self.kg_bias = nn.Parameter(torch.zeros(final_embedding_dim))
        self.text_embs_std = std_tokens

        # output projection layer with spectral normalization
        self.skip_to_final = nn.Linear(node_embedding_dim, final_embedding_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(node_embedding_dim, 4 * node_embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * node_embedding_dim, final_embedding_dim),
        )
        self.output_norm = nn.LayerNorm(final_embedding_dim)
        
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

        # Apply graph convolution layers
        for i, conv in enumerate(self.convs):
            device = self.convs[i].lin_query.weight.device
            dtype = self.convs[i].lin_query.weight.dtype
            x = x.to(device, dtype=dtype)
            edge_attr = edge_attr.to(device, dtype=dtype)
            edge_index = edge_index.to(device)

            x_in = x
            x = conv(x, edge_index, edge_attr)
            x = self.norms[i](x + x_in, graphs.batch.to(x.device))

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

        if self.codebook_dim > 0:
            x = self.downsample(x)
            x = self.downsample_norm(x)

        # ugly solution to problem of vector quantization not working with half precision
        x = x.float()
        self.vq.float()
        
        quantized, indices, commit_loss, all_codes = self.vq(x, return_all_codes=True)
        residual_loss = F.mse_loss(x, quantized)
        commit_loss = commit_loss.flatten().mean()
        loss = commit_loss * self.beta_commit + residual_loss

        dtype = self.kg_bias.dtype
        all_codes = all_codes.to(dtype=dtype)

        # apply the STE trick to get gradients to the encoder
        residual = x.to(dtype=dtype)
        tokens = []
        for i in range(self.num_quantizers):
            tokens.append( residual + (all_codes[i] - residual).detach())
            residual = residual - all_codes[i].detach()

        tokens = torch.stack(tokens) # (num_quantizers, B, dim)
        tokens = tokens.permute(1, 0, 2)  # (B, num_quantizers, dim)
        tokens = tokens.to(dtype=dtype)  # Ensure tokens have the correct dtype
        
        if self.codebook_dim > 0:
            tokens = self.upsample(tokens)
            tokens = self.upsample_norm(tokens)
        
        # in forward, after (optional) upsample:
        tokens = self.dropout(tokens)                             # [B, Q, node_D]
        proj = self.output_projection(tokens)                     # [B, Q, final_D]
        skip = self.skip_to_final(tokens)                         # [B, Q, final_D]
        tokens = self.output_norm(proj + skip)                    # residual in final space

        # per-batch standardization
        g_mu = tokens.mean(dim=(-2,-1), keepdim=True)
        g_std = tokens.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
        tokens = (tokens - g_mu) / g_std
        # match text emb std
        tokens = tokens * self.text_embs_std + self.kg_bias

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Output shape after adapter and normalization: {x.shape}")
            logging.debug(f"Quantized output shape: {all_codes.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
            logging.debug(f"Quantized output shape: {tokens.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
        
        # Return the quantized representations
        return tokens, indices, loss