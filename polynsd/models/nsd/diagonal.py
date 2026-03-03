import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, scatter


class DiagonalNSDConv(MessagePassing):
    def __init__(self, d, in_channels, hidden_dim, alpha=0.05):
        super().__init__(aggr="add", node_dim=0)
        self.d = d
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.map_generator = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),  # Output is a vector d
        )

    def forward(self, x_feat, x_stalk, edge_index):
        u_idx, v_idx = edge_index
        num_nodes = x_stalk.size(0)

        # 1. Normalization
        deg = degree(v_idx, num_nodes, dtype=x_stalk.dtype)
        self.norm = deg.pow(-0.5).clamp(min=0, max=1e6)

        # 2. Generate Diagonal Vectors
        w_v = self.map_generator(
            torch.cat([x_feat[v_idx], x_feat[u_idx]], dim=-1)
        ).view(-1, self.d, 1)
        w_u = self.map_generator(
            torch.cat([x_feat[u_idx], x_feat[v_idx]], dim=-1)
        ).view(-1, self.d, 1)

        # 3. Diagonal Part: sum(s_norm * w_v^2)
        s_norm = self.norm[v_idx] * self.norm[u_idx]
        weighted_w_v_sq = s_norm.view(-1, 1, 1) * (w_v**2)
        diag_blocks = scatter(
            weighted_w_v_sq, v_idx, dim=0, dim_size=num_nodes, reduce="sum"
        )

        # 4. Message: s_norm * (w_v * w_u) * h_u
        T_vec = w_v * w_u
        adj_out = self.propagate(edge_index, x=x_stalk, T_vec=T_vec, s_norm=s_norm)

        return x_stalk + self.alpha * (adj_out - diag_blocks * x_stalk)

    def message(self, x_j, T_vec, s_norm):
        return s_norm.view(-1, 1, 1) * T_vec * x_j
