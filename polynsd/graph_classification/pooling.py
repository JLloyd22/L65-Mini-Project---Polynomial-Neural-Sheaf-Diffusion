import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import softmax


class AttentionPooling(nn.Module):
    """Attention-based graph pooling."""
    def __init__(self, hidden_channels: int, heads: int = 1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, heads, bias=False)
        )
        self.heads = heads
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, hidden_channels]
        # batch: [num_nodes] indicating which graph each node belongs to
        
        # Compute attention scores
        attn_scores = self.attention(x)  # [num_nodes, heads]
        
        # Softmax per graph
        attn_weights = softmax(attn_scores, batch, dim=0)  # [num_nodes, heads]
        
        # Weighted sum per graph
        if self.heads == 1:
            attn_weights = attn_weights.squeeze(-1)  # [num_nodes]
            # Weighted aggregation
            out = scatter(x * attn_weights.unsqueeze(-1), batch, dim=0, reduce='sum')
        else:
            # Multi-head: average over heads
            x_expanded = x.unsqueeze(1).expand(-1, self.heads, -1)  # [num_nodes, heads, hidden]
            weighted = x_expanded * attn_weights.unsqueeze(-1)  # [num_nodes, heads, hidden]
            out = scatter(weighted, batch, dim=0, reduce='sum')  # [num_graphs, heads, hidden]
            out = out.mean(dim=1)  # [num_graphs, hidden]
        
        return out