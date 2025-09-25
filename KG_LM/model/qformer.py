import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class QFormerPool(nn.Module):
    """
    Q-Former architecture to pool graph representations using cross-attention with learnable query tokens.
    """
    def __init__(self, channels, num_heads, dropout=0.1):
        super(QFormerPool, self).__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 8, channels))  # 8 learnable query tokens
        self.cross_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        
        
    def forward(self, x, mask=None):
        """
        Forward pass of the Q-Former pooling layer.
        
        Args:
            x (torch.Tensor): Input graph node features of shape (batch_size, num_nodes, in_channels).
            mask (torch.Tensor, optional): Mask for the input nodes, shape (batch_size, num_nodes). Defaults to None.

        Returns:
            torch.Tensor: Pooled graph representation of shape (batch_size, out_channels).
        """
        logging.debug(f"QFormerPool input shape: %s", x.shape)
        
        # Apply cross-attention with query tokens
        central_node = x[:, 0, :] # (batch_size, in_channels)
        
        # convert mask to (batch_size*num_heads, query_tokens, num_nodes)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.query_tokens.size(1), -1)  # (batch_size, 8, num_nodes)
            mask = ~mask.repeat_interleave(self.cross_attention.num_heads, dim=0)  # (batch_size*num_heads, 8, num_nodes)

        x = self.cross_attention(self.query_tokens.expand(x.size(0), -1, -1), x, x, attn_mask=mask)[0]  # (batch_size, 8, in_channels)
        
        logging.debug(f"QFormerPool after cross-attention shape: %s", x.shape)
        
        x = x.mean(dim=1)  # (batch_size, in_channels)
        x = x + central_node  # Residual connection
        x = self.norm(x)
        
        logging.debug(f"QFormerPool output shape: %s", x.shape)
        
        
        return x
