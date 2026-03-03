import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, scatter


class GeneralNSDConv(MessagePassing):
    def __init__(self, d, in_channels, hidden_dim, alpha=0.05):
        super().__init__(aggr="add", node_dim=0)
        self.d = d
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Symmetric MLP to generate restriction maps (Full d x d)
        self.map_generator = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d * d),
        )

    def forward(self, x_feat, x_stalk, edge_index):
        # 1. Compute Symmetric Normalization Coefficients
        u_idx, v_idx = edge_index  # Source u, Target v
        num_nodes = x_stalk.size(0)

        deg = degree(v_idx, num_nodes, dtype=x_stalk.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self.norm = deg_inv_sqrt

        # 2. Generate Dynamic Restriction Maps
        feat_v, feat_u = x_feat[v_idx], x_feat[u_idx]

        # Symmetrically generate W_i (target) and W_j (source)
        W_i = self.map_generator(torch.cat([feat_v, feat_u], dim=-1)).view(
            -1, self.d, self.d
        )
        W_j = self.map_generator(torch.cat([feat_u, feat_v], dim=-1)).view(
            -1, self.d, self.d
        )

        # 3. Optimized Diagonal Pre-computation
        s_norm_edges = self.norm[v_idx] * self.norm[u_idx]
        W_i_sq = torch.matmul(W_i.transpose(-2, -1), W_i)
        weighted_W_i_sq = s_norm_edges.view(-1, 1, 1) * W_i_sq

        # diag_blocks results in [N, d, d]
        diag_blocks = scatter(
            weighted_W_i_sq, v_idx, dim=0, dim_size=num_nodes, reduce="sum"
        )
        diag_part = torch.matmul(diag_blocks, x_stalk)

        # 4. Propagate Adjacency Part: W_i^T @ W_j @ h_j
        T = torch.matmul(W_i.transpose(-2, -1), W_j)
        adj_out = self.propagate(edge_index, x=x_stalk, T=T, s_norm=s_norm_edges)

        # Final Bodnar Update: h = h + alpha * (Adj - Diag)
        return x_stalk + self.alpha * (adj_out - diag_part)

    def message(self, x_j, T, s_norm):
        # x_j: [E, d, 1] (Source stalk)
        # T: [E, d, d] (Transport: W_i^T @ W_j)
        transported = torch.matmul(T, x_j)
        return s_norm.view(-1, 1, 1) * transported
