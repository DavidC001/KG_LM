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


class StarGraphStructureDecoder(nn.Module):
    """
    Decodes star graph structure from central node embedding.
    Specifically designed for star graphs where all edges connect to a central node.
    """
    def __init__(self, node_dim: int, max_neighbors: int = 49, num_edge_types: int = 1000):
        super().__init__()
        self.node_dim = node_dim
        self.max_neighbors = max_neighbors  # max_nodes - 1 (excluding central node)
        self.num_edge_types = num_edge_types
        
        # Central node feature projector
        self.central_node_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Neighbor node generator from central node
        self.neighbor_generator = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, max_neighbors * node_dim)
        )
        
        # Edge type predictor for each neighbor
        self.edge_type_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),  # central + neighbor features
            nn.ReLU(),
            nn.Linear(node_dim, num_edge_types)
        )
        
        # Number of neighbors predictor
        self.num_neighbors_predictor = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, max_neighbors + 1)  # 0 to max_neighbors
        )
        
    def forward(self, central_embedding: torch.Tensor):
        """
        Predict star graph structure from central node embedding.
        
        Args:
            central_embedding: [B, node_dim] central node embedding
            
        Returns:
            dict containing:
                - central_node_features: [B, node_dim] refined central node features
                - neighbor_features: [B, max_neighbors, node_dim] neighbor node features
                - edge_types: [B, max_neighbors, num_edge_types] edge type logits
                - num_neighbors: [B, max_neighbors + 1] number of neighbors prediction
        """
        batch_size = central_embedding.shape[0]
        
        # Refine central node features
        central_features = self.central_node_proj(central_embedding)  # [B, node_dim]
        
        # Generate neighbor features
        neighbor_flat = self.neighbor_generator(central_embedding)  # [B, max_neighbors * node_dim]
        neighbor_features = neighbor_flat.view(batch_size, self.max_neighbors, self.node_dim)  # [B, max_neighbors, node_dim]
        
        # Predict number of neighbors
        num_neighbors_logits = self.num_neighbors_predictor(central_embedding)  # [B, max_neighbors + 1]
        
        # Predict edge types for each potential neighbor
        # Combine central and neighbor features for edge type prediction
        central_expanded = central_features.unsqueeze(1).expand(-1, self.max_neighbors, -1)  # [B, max_neighbors, node_dim]
        edge_input = torch.cat([central_expanded, neighbor_features], dim=-1)  # [B, max_neighbors, 2*node_dim]
        edge_type_logits = self.edge_type_predictor(edge_input)  # [B, max_neighbors, num_edge_types]
        
        return {
            'central_node_features': central_features,
            'neighbor_features': neighbor_features,
            'edge_types': edge_type_logits,
            'num_neighbors': num_neighbors_logits
        }


class KGDecoder(nn.Module):
    """
    Knowledge Graph Decoder that reconstructs star graph data from quantized representations.
    
    This decoder performs the inverse operation of KGEncoder, taking quantized tokens
    and attempting to reconstruct the original star graph structure and node features.
    
    Optimized for star graphs where node 0 is the central node and all other nodes
    are neighbors connected only to the central node.
    """
    def __init__(
        self,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        final_embedding_dim: int,
        dropout: float = 0.2,
        num_heads: int = 1,
        gat_layers: int = 3,
        graph_pooling: bool = True,
        num_quantizers: int = 3,
        codebook_size: int = 512,
        codebook_dim: int = 0,
        shared_codebook: bool = False,
        max_nodes: int = 50,
        num_edge_types: int = 1000,
        reconstruction_weight: float = 1.0,
        structure_weight: float = 0.1
    ):
        """
        Initialize the KG Decoder.
        
        Args:
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings  
            final_embedding_dim: Dimension of the final embeddings (from LLM)
            dropout: Dropout rate
            num_heads: Number of attention heads in GAT layers
            gat_layers: Number of GAT layers
            graph_pooling: Whether the encoder used graph pooling
            num_quantizers: Number of quantizers in the VQ
            codebook_size: Size of the codebook
            codebook_dim: Dimension of the codebook (if different from node_embedding_dim)
            shared_codebook: Whether to use shared codebook
            max_nodes: Maximum number of nodes in a graph
            num_edge_types: Number of different edge types to predict
            reconstruction_weight: Weight for reconstruction loss
            structure_weight: Weight for graph structure loss
        """
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.graph_pooling = graph_pooling
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.max_nodes = max_nodes
        self.reconstruction_weight = reconstruction_weight
        self.structure_weight = structure_weight
        
        # Reverse the encoder's final projection
        self.input_projection = nn.Sequential(
            nn.Linear(final_embedding_dim, 4 * node_embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * node_embedding_dim, node_embedding_dim),
        )
        self.input_norm = nn.LayerNorm(node_embedding_dim)
        self.skip_from_final = nn.Linear(final_embedding_dim, node_embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Reverse codebook dimension projection if used
        if codebook_dim > 0:
            self.upsample_reverse = nn.Linear(node_embedding_dim, codebook_dim)
            self.upsample_norm_reverse = nn.LayerNorm(codebook_dim)
            self.downsample_reverse = nn.Linear(codebook_dim, node_embedding_dim)
            self.downsample_norm_reverse = nn.LayerNorm(node_embedding_dim)
            
        # Store reference to encoder's VQ for proper reconstruction
        self.encoder_vq = None
        
        # Star graph structure decoder (if not using pooling)
        if not graph_pooling:
            self.structure_decoder = StarGraphStructureDecoder(
                node_dim=node_embedding_dim,
                max_neighbors=max_nodes - 1,  # Exclude central node
                num_edge_types=num_edge_types
            )
        
        # GAT layers for graph reconstruction (reverse of encoder)
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
        
        # Final output projections
        self.node_output_proj = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.edge_output_proj = nn.Linear(edge_embedding_dim, edge_embedding_dim)
        
    def forward(
        self, 
        quantized_tokens: torch.Tensor,
        quantized_indices: torch.Tensor,
        original_graphs: Batch = None,
        target_node_features: torch.Tensor = None,
        target_edge_features: torch.Tensor = None
    ):
        """
        Decode quantized representations back to graph data.
        
        Args:
            quantized_tokens: [B, Q, final_embedding_dim] quantized token representations
            quantized_indices: [B, Q] quantized indices
            original_graphs: Original graph batch for structure reference (optional)
            target_node_features: Target node features for loss calculation (optional)
            target_edge_features: Target edge features for loss calculation (optional)
            
        Returns:
            dict containing:
                - reconstructed_node_features: Reconstructed node features
                - reconstructed_edge_features: Reconstructed edge features (if available)
                - predicted_structure: Predicted graph structure (if not pooling)
                - reconstruction_loss: Total reconstruction loss
        """
        batch_size, num_quantizers, _ = quantized_tokens.shape
        
        logging.debug(f"Decoding quantized tokens with shape: {quantized_tokens.shape}")
        
        # Use the quantized tokens directly - they already contain the encoded information
        # Average across quantization levels to get a single representation per graph
        averaged_tokens = quantized_tokens.mean(dim=1)  # [B, final_embedding_dim]
        
        # Reverse the final embedding projection (inverse of encoder's output projection)
        proj_reversed = self.input_projection(averaged_tokens)  # [B, node_embedding_dim]
        skip_reversed = self.skip_from_final(averaged_tokens)   # [B, node_embedding_dim]
        node_embeddings = self.input_norm(proj_reversed + skip_reversed)  # [B, node_embedding_dim]
        
        node_embeddings = self.dropout(node_embeddings)
        
        # Reverse codebook dimension changes if applied
        if self.codebook_dim > 0:
            node_embeddings = self.upsample_reverse(node_embeddings)
            node_embeddings = self.upsample_norm_reverse(node_embeddings)
        
        # The final embeddings represent the central node of the star graph
        central_node_embeddings = node_embeddings  # [B, node_embedding_dim or codebook_dim]
        
        reconstruction_loss = 0.0
        reconstructed_node_features = None
        reconstructed_edge_features = None
        predicted_structure = None
        
        if self.graph_pooling:
            # If pooling was used, we only have a single embedding per graph
            # We can reconstruct individual node features if target is provided
            if target_node_features is not None:
                # Simple approach: use the central node embedding as a prototype for all nodes
                # In a more sophisticated approach, you might want to learn to unpool
                reconstructed_node_features = central_node_embeddings.unsqueeze(1).expand(-1, target_node_features.shape[1], -1)
                reconstructed_node_features = self.node_output_proj(reconstructed_node_features)
                
                # Calculate reconstruction loss
                reconstruction_loss += F.mse_loss(reconstructed_node_features, target_node_features)
        else:
            # For star graphs without pooling, reconstruct the star structure
            # The central_node_embeddings represents the central node
            
            # Predict star graph structure using the central node embedding
            if hasattr(self, 'structure_decoder'):
                star_structure = self.structure_decoder(central_node_embeddings)
                predicted_structure = star_structure
                
                # Reconstruct node features: central node + neighbors
                central_features = star_structure['central_node_features']  # [B, node_dim]
                neighbor_features = star_structure['neighbor_features']  # [B, max_neighbors, node_dim]
                
                # Combine central and neighbor features to form full node features
                # Central node is at index 0, neighbors follow
                reconstructed_node_features = torch.cat([
                    central_features.unsqueeze(1),  # [B, 1, node_dim]
                    neighbor_features  # [B, max_neighbors, node_dim]
                ], dim=1)  # [B, max_nodes, node_dim]
                
                # Apply final projection
                reconstructed_node_features = self.node_output_proj(reconstructed_node_features)
            else:
                # Fallback: expand the central embedding to all nodes
                expanded_embeddings = central_node_embeddings.unsqueeze(1).expand(-1, self.max_nodes, -1)
                reconstructed_node_features = self.node_output_proj(expanded_embeddings)
            
            # Calculate reconstruction loss if targets are provided
            if target_node_features is not None:
                # For star graphs, target should have central node first, then neighbors
                target_size = target_node_features.shape[1]
                if target_size <= self.max_nodes:
                    recon_features = reconstructed_node_features[:, :target_size, :]
                else:
                    # If target has more nodes than max, truncate target
                    target_node_features = target_node_features[:, :self.max_nodes, :]
                    recon_features = reconstructed_node_features
                
                reconstruction_loss += F.mse_loss(recon_features, target_node_features)
                
                # Additional loss for star graph structure if available
                if hasattr(self, 'structure_decoder') and predicted_structure is not None:
                    # Loss on number of neighbors prediction
                    if original_graphs is not None:
                        # Extract actual number of neighbors from original graphs
                        # This would require processing the batch to get neighbor counts
                        pass  # TODO: Implement if needed for training
        
        # Reconstruct edge features if targets are provided
        if target_edge_features is not None:
            # Simple approach: use central node embedding as prototype for edges
            num_edges = target_edge_features.shape[1]
            reconstructed_edge_features = central_node_embeddings.unsqueeze(1).expand(-1, num_edges, -1)
            reconstructed_edge_features = self.edge_output_proj(reconstructed_edge_features)
            
            reconstruction_loss += F.mse_loss(reconstructed_edge_features, target_edge_features)
        
        total_loss = self.reconstruction_weight * reconstruction_loss
        
        logging.debug(f"Reconstruction loss: {reconstruction_loss:.4f}")
        
        return {
            'reconstructed_node_features': reconstructed_node_features,
            'reconstructed_edge_features': reconstructed_edge_features,
            'predicted_structure': predicted_structure,
            'reconstruction_loss': total_loss,
            'central_embeddings': central_node_embeddings
        }
    
    def set_encoder_vq(self, encoder_vq):
        """
        Set the vector quantizer from the encoder to ensure we use the same codebooks.
        This is crucial for proper reconstruction, though we primarily use the tokens directly.
        """
        self.encoder_vq = encoder_vq
        logging.debug("Set encoder VQ reference in decoder")
            
    def construct_star_graph_edges(self, num_neighbors: torch.Tensor, batch_size: int, device: torch.device):
        """
        Construct edge indices for star graphs based on predicted number of neighbors.
        
        Args:
            num_neighbors: [B] tensor of number of neighbors for each graph
            batch_size: batch size
            device: device to place tensors on
            
        Returns:
            edge_index: [2, total_edges] edge indices for all graphs in batch
            batch: [total_nodes] batch assignment for each node
        """
        edge_indices = []
        node_offsets = []
        batch_assignments = []
        current_offset = 0
        
        for b in range(batch_size):
            n_neighbors = num_neighbors[b].item()
            n_nodes = n_neighbors + 1  # +1 for central node
            
            # Central node is at index 0 (relative to current graph)
            # Neighbors are at indices 1, 2, ..., n_neighbors
            
            # Create edges: central -> neighbors and neighbors -> central (bidirectional)
            if n_neighbors > 0:
                central_idx = current_offset
                neighbor_indices = torch.arange(1, n_neighbors + 1, device=device) + current_offset
                
                # Edges from central to neighbors
                edges_out = torch.stack([
                    torch.full((n_neighbors,), central_idx, device=device),
                    neighbor_indices
                ])
                
                # Edges from neighbors to central
                edges_in = torch.stack([
                    neighbor_indices,
                    torch.full((n_neighbors,), central_idx, device=device)
                ])
                
                # Combine both directions
                graph_edges = torch.cat([edges_out, edges_in], dim=1)
                edge_indices.append(graph_edges)
            
            # Track node offsets and batch assignments
            node_offsets.append(current_offset)
            batch_assignments.extend([b] * n_nodes)
            current_offset += n_nodes
        
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
        batch = torch.tensor(batch_assignments, dtype=torch.long, device=device)
        
        return edge_index, batch
