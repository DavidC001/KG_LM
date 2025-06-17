"""
pytorch model that when received a Batch of graphs,
returns for each the residual vector quantized representsion for each graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# graph pooling
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import Batch
from vector_quantize_pytorch import ResidualVQ


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
        
        # adapter layer
        self.adapter = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # normalization layer
        self.norm = nn.LayerNorm(node_embedding_dim)

        # residual vector quantization layer
        self.vq = ResidualVQ(
            dim=node_embedding_dim,
            num_quantizers= num_quantizers,
            codebook_size=codebook_size,
            shared_codebook= shared_codebook,
        )
        
        # output projection layer
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
        x, edge_index, edge_attr = graphs.x, graphs.edge_index, graphs.edge_attr
        
        # Ensure input tensors have the same dtype as model parameters
        model_dtype = next(self.parameters()).dtype
        x = x.to(dtype=model_dtype)
        if edge_attr is not None:
            edge_attr = edge_attr.to(dtype=model_dtype)
        
        # Apply GATv2Conv
        x = self.conv(x, edge_index, edge_attr)
        
        if self.graph_pooling:
            x = global_mean_pool(x, graphs.batch)
        else:
            # Find the first occurrence of each batch index
            num_graphs = graphs.batch.max().item() + 1
            
            # Find first node index for each graph in the batch (GPU-optimized, no loops)
            first_indices = torch.zeros(num_graphs, dtype=torch.long, device=graphs.batch.device)
            mask = graphs.batch[:-1] != graphs.batch[1:] # Find where the batch index changes
            first_indices[1:] = torch.where(mask)[0] + 1 # Shift indices by 1 to account for the first node in each graph
            
            x = x[first_indices]
            
        # Apply adapter layer
        x = self.adapter(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Normalize the output
        x = self.norm(x)
        
        # Ensure tensor is in the correct dtype before vector quantization
        x = x.to(dtype=model_dtype)
        
        # Apply residual vector quantization
        _, indices, loss, quantized_x = self.vq(x, return_all_codes = True)
        
        # invert first dimension (num_quantizers) with second dimension (batch size)
        quantized_x = quantized_x.permute(1, 0, 2)
        
        quantized_x = self.output_projection(quantized_x)
        
        # Return the quantized representations
        return quantized_x, indices, loss