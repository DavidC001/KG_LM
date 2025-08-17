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
        node_embedding_dim, 
        edge_embedding_dim,
        final_embedding_dim,
        dropout=0.2,
        num_heads=1,
        num_quantizers=3,
        codebook_size=512,
        codebook_dim=0,
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
            codebook_dim (int): Dimension of the downsampled codebook. If 0, no downsampling is applied.
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
            dropout=dropout,
        )
        self.num_heads = num_heads
        
        # adapter layer with spectral normalization for stability
        self.adapter = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_quantizers = num_quantizers

        # normalization layer
        self.norm = nn.LayerNorm(node_embedding_dim)
        self.out_norm = nn.LayerNorm(final_embedding_dim)

        
        self.vq = ResidualVQ(
            dim=node_embedding_dim,
            codebook_dim=codebook_dim if codebook_dim > 0 else None,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            shared_codebook=shared_codebook,
            threshold_ema_dead_code = 2
        )

        # output projection layer with spectral normalization
        self.output_projection = nn.Linear(node_embedding_dim, final_embedding_dim)
        
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
        
        x = x.to(self.conv.lin_l.weight.device)
        edge_index = edge_index.to(self.conv.lin_l.weight.device)
        edge_attr = edge_attr.to(self.conv.lin_l.weight.device)
        
        logging.debug("Moved tensors")

        # edge and node dropout
        if self.training:
            edge_attr = self.dropout(edge_attr)
            x = self.dropout(x)

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
        # Normalize the output
        x = self.norm(x)

        # ugly solution to problem of vector quantization not working with half precision
        x = x.float()
        self.vq.float()
        
        quantized, indices, commit_loss, all_codes = self.vq(x, return_all_codes=True)
        residual_loss = F.mse_loss(x, quantized)
        commit_loss = commit_loss.flatten().mean()
        loss = commit_loss + residual_loss

        dtype = self.output_projection.weight.dtype

        all_codes = all_codes.to(dtype=dtype)

        # apply the STE trick to get gradients to the encoder
        residual = x.to(dtype=dtype)
        tokens = []
        for i in range(self.num_quantizers):
            tokens.append( x + (all_codes[i] - x).detach())
            residual = residual - all_codes[i].detach()

        tokens = torch.stack(tokens) # (num_quantizers, B, dim)
        tokens = tokens.permute(1, 0, 2)  # (B, num_quantizers, dim)
        tokens = tokens.to(dtype=dtype)  # Ensure tokens have the correct dtype
        tokens = self.dropout(tokens)
        tokens = self.output_projection(tokens)
        
        # Normalize the quantized output
        tokens = self.out_norm(tokens)
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Output shape after adapter and normalization: {x.shape}")
            logging.debug(f"Quantized output shape: {all_codes.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
            logging.debug(f"Quantized output shape: {tokens.shape}, Indices shape: {indices.shape}, Loss: {loss.item()}")
        
        # Return the quantized representations
        return tokens, indices, loss