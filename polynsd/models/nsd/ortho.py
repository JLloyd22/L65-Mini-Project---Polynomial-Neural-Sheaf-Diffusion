import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, scatter


class OrthogonalNSDConv(MessagePassing):
    def __init__(self, d, in_channels, hidden_dim, alpha=0.05, clamp_val=10.0):
        super().__init__(aggr="add", node_dim=0)
        self.d = d
        self.clamp_val = clamp_val
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.num_params = (d * (d - 1)) // 2

        self.map_generator = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_params),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.map_generator:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.001)
                nn.init.constant_(m.bias, 0.0)

    def cayley(self, params):
        # 1. Clamp params to avoid extreme rotations and numerical instability
        params = torch.clamp(params, -self.clamp_val, self.clamp_val)

        batch_size = params.size(0)
        A = torch.zeros(batch_size, self.d, self.d, device=params.device)
        indices = torch.triu_indices(self.d, self.d, offset=1)
        A[:, indices[0], indices[1]] = params
        A = A - A.transpose(-2, -1)

        # 2. Add a small epsilon to the diagonal for inversion stability
        eye = torch.eye(self.d, device=params.device).unsqueeze(0)
        return torch.linalg.solve(eye - A, eye + A)

    def forward(self, x_feat, x_stalk, edge_index):
        u_idx, v_idx = edge_index
        num_nodes = x_stalk.size(0)

        deg = degree(v_idx, num_nodes, dtype=x_stalk.dtype)
        norm = deg.pow(-0.5).clamp(min=0, max=1e6)
        norm[torch.isinf(norm)] = 0

        # Generate maps
        params_v = self.map_generator(torch.cat([x_feat[v_idx], x_feat[u_idx]], dim=-1))
        params_u = self.map_generator(torch.cat([x_feat[u_idx], x_feat[v_idx]], dim=-1))

        # Apply gradient clipping hook to params
        if params_v.requires_grad:
            params_v.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
            params_u.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        W_v, W_u = self.cayley(params_v), self.cayley(params_u)

        s_norm = norm[v_idx] * norm[u_idx]
        diag_coeffs = scatter(s_norm, v_idx, dim=0, dim_size=num_nodes, reduce="sum")
        diag_part = diag_coeffs.view(-1, 1, 1) * x_stalk

        T = torch.matmul(W_v.transpose(-2, -1), W_u)
        adj_out = self.propagate(edge_index, x=x_stalk, T=T, s_norm=s_norm)

        return x_stalk + self.alpha * (adj_out - diag_part)

    def message(self, x_j, T, s_norm):
        return s_norm.view(-1, 1, 1) * torch.matmul(T, x_j)
