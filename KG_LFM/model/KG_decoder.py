"""
KG Decoder that reconstructs star graph representations from quantized tokens.
This module provides the inverse operation of the KG encoder, taking quantized 
representations and attempting to reconstruct the original star graph structure and features.

Optimized for star graphs where:
- One central node is connected to multiple neighbor nodes
- All edges go from/to the central node (no neighbor-to-neighbor edges)
- Node at index 0 is always the central node
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
from torch_geometric.data import Batch, Data
import logging

class KGDecoder(nn.Module):
    """
    Knowledge Graph Decoder that reconstructs star graph data from quantized representations.
    
    This decoder performs the inverse operation of KGEncoder, taking quantized tokens
    and attempting to reconstruct the original star graph structure and node features.
    
    Optimized for star graphs where node 0 is the central node and all other nodes
    are neighbors connected only to the central node.
    
    The decoder supports two reconstruction modes:
    1. Node feature reconstruction: Reconstructs node embeddings using L2 loss
    2. Graph structure prediction: Predicts graph topology and edge types
    """
    def __init__(
        self,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        final_embedding_dim: int,
        dropout: float = 0.2,
        num_heads: int = 1,
        decoder_layers: int = 2,
        max_nodes: int = 50,
        num_edge_types: int = 1000,
        reconstruction_weight: float = 1.0,
        structure_weight: float = 0.1,
        num_quantizers: int = 3,
    ):
        """
        Initialize the KG Decoder.
        
        Args:
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings  
            final_embedding_dim: Dimension of final LLM embeddings
            dropout: Dropout rate
            num_heads: Number of attention heads for graph layers
            decoder_layers: Number of decoder graph layers
            max_nodes: Maximum number of nodes to reconstruct
            num_edge_types: Number of possible edge types
            reconstruction_weight: Weight for node reconstruction loss
            structure_weight: Weight for structure prediction loss
            num_quantizers: Number of quantizers (must match encoder)
        """
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.reconstruction_weight = reconstruction_weight
        self.structure_weight = structure_weight
        self.num_quantizers = num_quantizers
        
        # Input projection: from final_embedding_dim back to node_embedding_dim
        self.input_projection = nn.Sequential(
            nn.Linear(final_embedding_dim, 2 * node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
        )
        self.input_norm = nn.LayerNorm(node_embedding_dim)
        
        # Quantizer aggregation: combine multiple quantized tokens
        self.quantizer_attention = nn.MultiheadAttention(
            embed_dim=node_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.quantizer_norm = nn.LayerNorm(node_embedding_dim)
        
        # Graph structure decoder
        self.decoder_layers = nn.ModuleList([
            TransformerConv(
                in_channels=node_embedding_dim,
                out_channels=node_embedding_dim // num_heads,
                heads=num_heads,
                edge_dim=edge_embedding_dim,
                dropout=dropout,
                concat=True
            )
            for _ in range(decoder_layers)
        ])
        self.decoder_norms = nn.ModuleList([
            GraphNorm(node_embedding_dim) for _ in range(decoder_layers)
        ])
        
        # Node feature reconstruction head
        self.node_reconstruction_head = nn.Sequential(
            nn.Linear(node_embedding_dim, 2 * node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
        )
        
        # Star graph structure prediction heads
        self.num_nodes_predictor = nn.Sequential(
            nn.Linear(node_embedding_dim, node_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_embedding_dim // 2, max_nodes),
        )
        
        self.edge_type_predictor = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_embedding_dim, num_edge_types),
        )
        
        # Central node feature predictor (for star graphs)
        self.central_node_predictor = nn.Sequential(
            nn.Linear(node_embedding_dim, 2 * node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
        )
        
        # Neighbor node feature predictor
        self.neighbor_predictor = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, 2 * node_embedding_dim),  # central + pooled info
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Reference to encoder's vector quantizer (set externally)
        self.encoder_vq = None
        
    def set_encoder_vq(self, encoder_vq):
        """Set reference to encoder's vector quantizer for reconstruction."""
        self.encoder_vq = encoder_vq
        
    def forward(
        self, 
        quantized_tokens: torch.Tensor,
        original_graphs: Batch = None,
        reconstruct_structure: bool = True,
    ):
        """
        Reconstruct graphs from quantized tokens.
        
        Args:
            quantized_tokens: [B, Q, final_embedding_dim] quantized representations
            original_graphs: Original graph batch for reference (optional)
            reconstruct_structure: Whether to predict graph structure
            
        Returns:
            dict: Reconstruction results containing:
                - reconstructed_graphs: List of reconstructed Data objects
                - reconstruction_loss: L2 loss for node features
                - structure_loss: Loss for structure prediction (if enabled)
                - total_loss: Combined weighted loss
        """
        batch_size, num_quantizers, embed_dim = quantized_tokens.shape
        device = quantized_tokens.device
        
        # Project from final embedding space back to node embedding space
        # [B, Q, final_D] -> [B, Q, node_D]
        projected_tokens = self.input_projection(quantized_tokens)
        projected_tokens = self.input_norm(projected_tokens)
        
        # Aggregate information across quantizers using attention
        # Use each quantizer token as query, key, value
        aggregated, _ = self.quantizer_attention(
            projected_tokens, projected_tokens, projected_tokens
        )
        aggregated = self.quantizer_norm(aggregated + projected_tokens)
        
        # Pool across quantizers to get single graph representation [B, node_D]
        graph_repr = aggregated.mean(dim=1)  # Average pooling across quantizers
        
        results = {
            'reconstruction_loss': torch.tensor(0.0, device=device),
            'structure_loss': torch.tensor(0.0, device=device),
            'total_loss': torch.tensor(0.0, device=device),
            'reconstructed_graphs': []
        }
        
        if original_graphs is not None:
            # Reconstruct node features for each graph in batch
            reconstruction_loss = self._reconstruct_node_features(
                graph_repr, original_graphs
            )
            results['reconstruction_loss'] = reconstruction_loss
            
            if reconstruct_structure:
                # Predict graph structure
                structure_loss = self._reconstruct_structure(
                    graph_repr, original_graphs
                )
                results['structure_loss'] = structure_loss
            
            # Generate reconstructed graphs
            reconstructed_graphs = self._generate_reconstructed_graphs(
                graph_repr, original_graphs
            )
            results['reconstructed_graphs'] = reconstructed_graphs
            
            # Compute total weighted loss
            total_loss = (
                self.reconstruction_weight * reconstruction_loss +
                self.structure_weight * structure_loss
            )
            results['total_loss'] = total_loss
        
        return results
    
    def _reconstruct_node_features(self, graph_repr: torch.Tensor, original_graphs: Batch):
        """Reconstruct node features using L2 loss."""
        batch_size = graph_repr.shape[0]
        device = graph_repr.device
        
        reconstruction_loss = torch.tensor(0.0, device=device)
        
        # Process each graph in the batch
        for i in range(batch_size):
            # Get original graph nodes for this batch item
            mask = original_graphs.batch == i
            original_nodes = original_graphs.x[mask]  # [num_nodes_i, node_dim]
            num_nodes = original_nodes.shape[0]
            
            if num_nodes == 0:
                continue
                
            # Predict central node (index 0 in star graph)
            central_pred = self.central_node_predictor(graph_repr[i:i+1])  # [1, node_dim]
            
            # Predict neighbor nodes
            neighbor_preds = []
            if num_nodes > 1:
                # Replicate central info for each neighbor
                central_expanded = central_pred.expand(num_nodes - 1, -1)  # [num_neighbors, node_dim]
                graph_expanded = graph_repr[i:i+1].expand(num_nodes - 1, -1)  # [num_neighbors, node_dim]
                neighbor_input = torch.cat([central_expanded, graph_expanded], dim=-1)  # [num_neighbors, 2*node_dim]
                neighbor_preds = self.neighbor_predictor(neighbor_input)  # [num_neighbors, node_dim]
            
            # Combine predictions
            if num_nodes == 1:
                all_pred_nodes = central_pred
            else:
                all_pred_nodes = torch.cat([central_pred, neighbor_preds], dim=0)  # [num_nodes, node_dim]
            
            # Compute L2 reconstruction loss
            node_loss = F.mse_loss(all_pred_nodes, original_nodes)
            reconstruction_loss = reconstruction_loss + node_loss
        
        # Average over batch
        reconstruction_loss = reconstruction_loss / batch_size
        return reconstruction_loss
    
    def _reconstruct_structure(self, graph_repr: torch.Tensor, original_graphs: Batch):
        """Predict graph structure (number of nodes, edge types)."""
        batch_size = graph_repr.shape[0]
        device = graph_repr.device
        
        structure_loss = torch.tensor(0.0, device=device)
        
        # Predict number of nodes for each graph
        num_nodes_logits = self.num_nodes_predictor(graph_repr)  # [B, max_nodes]
        
        # Get true number of nodes for each graph in batch
        true_num_nodes = []
        for i in range(batch_size):
            mask = original_graphs.batch == i
            true_num_nodes.append(mask.sum().item())
        
        true_num_nodes = torch.tensor(true_num_nodes, device=device, dtype=torch.long)
        
        # Cross-entropy loss for number of nodes prediction
        # Clamp to valid range
        true_num_nodes = torch.clamp(true_num_nodes - 1, 0, self.max_nodes - 1)  # -1 because 0-indexed
        num_nodes_loss = F.cross_entropy(num_nodes_logits, true_num_nodes)
        
        structure_loss = structure_loss + num_nodes_loss
        
        return structure_loss
    
    def _generate_reconstructed_graphs(self, graph_repr: torch.Tensor, original_graphs: Batch):
        """Generate reconstructed graph Data objects."""
        batch_size = graph_repr.shape[0]
        reconstructed_graphs = []
        
        for i in range(batch_size):
            # Get original graph structure for reference
            mask = original_graphs.batch == i
            original_nodes = original_graphs.x[mask]
            num_nodes = original_nodes.shape[0]
            
            if num_nodes == 0:
                continue
            
            # Reconstruct nodes (same as in _reconstruct_node_features)
            central_pred = self.central_node_predictor(graph_repr[i:i+1])
            
            if num_nodes == 1:
                reconstructed_x = central_pred
            else:
                central_expanded = central_pred.expand(num_nodes - 1, -1)
                graph_expanded = graph_repr[i:i+1].expand(num_nodes - 1, -1)
                neighbor_input = torch.cat([central_expanded, graph_expanded], dim=-1)
                neighbor_preds = self.neighbor_predictor(neighbor_input)
                reconstructed_x = torch.cat([central_pred, neighbor_preds], dim=0)
            
            # Create star graph structure (central node connected to all others)
            if num_nodes == 1:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=graph_repr.device)
                edge_attr = torch.empty((0, self.edge_embedding_dim), device=graph_repr.device)
            else:
                # Create bidirectional edges between central (0) and all neighbors
                edges_out = torch.stack([
                    torch.zeros(num_nodes - 1, dtype=torch.long, device=graph_repr.device),  # from central
                    torch.arange(1, num_nodes, dtype=torch.long, device=graph_repr.device)   # to neighbors
                ], dim=0)
                edges_in = torch.stack([
                    torch.arange(1, num_nodes, dtype=torch.long, device=graph_repr.device),  # from neighbors  
                    torch.zeros(num_nodes - 1, dtype=torch.long, device=graph_repr.device)   # to central
                ], dim=0)
                edge_index = torch.cat([edges_out, edges_in], dim=1)  # [2, 2*(num_nodes-1)]
                
                # Use original edge attributes or predict them
                if hasattr(original_graphs, 'edge_attr') and original_graphs.edge_attr is not None:
                    # Extract edge attributes for this graph
                    edge_mask = torch.isin(original_graphs.edge_index[0], torch.where(mask)[0]) | \
                               torch.isin(original_graphs.edge_index[1], torch.where(mask)[0])
                    if edge_mask.any():
                        original_edge_attr = original_graphs.edge_attr[edge_mask]
                        if original_edge_attr.shape[0] >= edge_index.shape[1]:
                            edge_attr = original_edge_attr[:edge_index.shape[1]]
                        else:
                            # Repeat if not enough edges
                            repeats = edge_index.shape[1] // original_edge_attr.shape[0] + 1
                            edge_attr = original_edge_attr.repeat(repeats, 1)[:edge_index.shape[1]]
                    else:
                        edge_attr = torch.zeros((edge_index.shape[1], self.edge_embedding_dim), 
                                              device=graph_repr.device)
                else:
                    edge_attr = torch.zeros((edge_index.shape[1], self.edge_embedding_dim), 
                                          device=graph_repr.device)
            
            # Create reconstructed graph
            reconstructed_graph = Data(
                x=reconstructed_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes
            )
            
            reconstructed_graphs.append(reconstructed_graph)
        
        return reconstructed_graphs