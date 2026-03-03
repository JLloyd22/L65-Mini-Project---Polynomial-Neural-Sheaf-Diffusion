from typing import Optional, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_sparse
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax

# from Barbero et al.

# =====================================================================================
# SHEAF ADJACENCY BUILDER
# =====================================================================================


class SheafAdjacencyBuilder(nn.Module):
    """Builds sheaf adjacency matrix Â_F for SheafAN.

    Off-diagonal blocks: Â_F[i,j] = F^T_{i⊴e} F_{j⊴e} = P_{ij}
    Diagonal blocks: I_d (with self-loops)

    Args:
        num_nodes: Number of nodes in graph
        edge_index: Edge indices (2, num_edges) - should include both directions
        d: Stalk dimension
        add_self_loops: Whether to add identity self-loops
        normalised: Whether to apply symmetric normalization
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: Tensor,
        d: int,
        add_self_loops: bool = True,
        normalised: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.add_self_loops = add_self_loops
        self.normalised = normalised

        self._assign_buffer("edge_index", edge_index)
        self._precompute_indices()

    def _assign_buffer(self, name: str, tensor: Tensor):
        if name in self._buffers:
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor)

    def _precompute_indices(self):
        """Precompute sparse matrix index structure (vectorized)."""
        if self.edge_index is None:
            return

        row, col = self.edge_index
        device = self.edge_index.device
        num_edges = row.size(0)

        # Vectorized reverse edge lookup using searchsorted
        edge_hash = row * self.num_nodes + col
        reverse_hash = col * self.num_nodes + row

        sorted_hash, sort_idx = edge_hash.sort()
        reverse_positions = torch.searchsorted(sorted_hash, reverse_hash)
        reverse_positions = reverse_positions.clamp(max=num_edges - 1)
        reverse_idx = sort_idx[reverse_positions]

        # Handle missing reverse edges gracefully
        valid_reverse = sorted_hash[reverse_positions] == reverse_hash
        reverse_idx = torch.where(
            valid_reverse, reverse_idx, torch.arange(num_edges, device=device)
        )
        self._assign_buffer("reverse_edge_idx", reverse_idx.contiguous())

        # Use broadcasting instead of multiple expand() calls
        d_range = torch.arange(self.d, device=device)

        block_row_base = row * self.d
        block_col_base = col * self.d

        # Vectorized outer product for block indices
        off_diag_row = (block_row_base.view(-1, 1, 1) + d_range.view(1, -1, 1)).expand(
            -1, -1, self.d
        )
        off_diag_col = (block_col_base.view(-1, 1, 1) + d_range.view(1, 1, -1)).expand(
            -1, self.d, -1
        )

        self._assign_buffer("off_diag_row_idx", off_diag_row.reshape(-1).contiguous())
        self._assign_buffer("off_diag_col_idx", off_diag_col.reshape(-1).contiguous())

        if self.add_self_loops:
            diag_base = torch.arange(self.num_nodes, device=device) * self.d
            diag_row = (diag_base.view(-1, 1, 1) + d_range.view(1, -1, 1)).expand(
                -1, -1, self.d
            )
            diag_col = (diag_base.view(-1, 1, 1) + d_range.view(1, 1, -1)).expand(
                -1, self.d, -1
            )

            self._assign_buffer("diag_row_idx", diag_row.reshape(-1).contiguous())
            self._assign_buffer("diag_col_idx", diag_col.reshape(-1).contiguous())

            # Precompute identity values (constant)
            identity_vals = (
                torch.eye(self.d, device=device)
                .unsqueeze(0)
                .expand(self.num_nodes, -1, -1)
            )
            self._assign_buffer(
                "_identity_values", identity_vals.reshape(-1).contiguous()
            )

    def update_graph(self, num_nodes: int, edge_index: Tensor):
        self.num_nodes = num_nodes
        self._assign_buffer("edge_index", edge_index)
        self._precompute_indices()

    def forward(
        self,
        restriction_maps: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Build sheaf adjacency matrix from restriction maps.

        Args:
            restriction_maps: (num_edges, d, d) orthogonal restriction maps

        Returns:
            (indices, values): Sparse matrix representation
            transport_maps: (num_edges, d, d) P_{ij} = F_i^T F_j
        """
        device = restriction_maps.device
        dtype = restriction_maps.dtype

        # Transport maps: P_{ij} = F_i^T @ F_j
        F_src = restriction_maps
        F_dst = restriction_maps[self.reverse_edge_idx]
        transport_maps = torch.bmm(F_src.transpose(-2, -1), F_dst)

        # Flatten for sparse values
        off_diag_values = transport_maps.reshape(-1)
        off_diag_indices = torch.stack(
            [self.off_diag_row_idx, self.off_diag_col_idx], dim=0
        )

        if self.add_self_loops:
            # Use precomputed identity values (cast dtype if needed)
            diag_values = self._identity_values.to(dtype=dtype)
            diag_indices = torch.stack([self.diag_row_idx, self.diag_col_idx], dim=0)

            indices = torch.cat([off_diag_indices, diag_indices], dim=1)
            values = torch.cat([off_diag_values, diag_values], dim=0)
        else:
            indices = off_diag_indices
            values = off_diag_values

        if self.normalised:
            values = self._symmetric_norm(indices, values)

        return (indices, values), transport_maps

    def _symmetric_norm(self, indices: Tensor, values: Tensor) -> Tensor:
        """Apply symmetric normalization D^{-1/2} A D^{-1/2}."""
        N = self.num_nodes * self.d
        row, col = indices

        deg = torch.zeros(N, device=values.device, dtype=values.dtype)
        deg.scatter_add_(0, row, values.abs())

        # In-place operations for efficiency
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

        return deg_inv_sqrt[row] * values * deg_inv_sqrt[col]


# =====================================================================================
# GAT-STYLE ATTENTION
# =====================================================================================

from typing import Optional, Union, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import softmax as pyg_softmax


class SheafAttention(nn.Module):
    """GAT-style scalar attention for SheafAN.

    If concatenate=False (default, additive):
        α_{ij} = softmax_j(LeakyReLU(α_src[i] + α_dst[j] + α_edge[e]))

    If concatenate=True (matches sheaf learner encoding):
        α_{ij} = softmax_j(LeakyReLU(a^T [x_i || x_j || t_src || t_dst || t_edge]))
    """

    USES_NODE_TYPE = {"type_concat", "node_type_concat", "types_only", "node_type"}
    USES_EDGE_TYPE = {
        "type_concat",
        "edge_type_concat",
        "types_only",
        "edge_type",
        "type_ensemble",
    }
    NO_FEATURES = {"types_only", "node_type", "edge_type"}

    def __init__(
        self,
        in_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        learner_type: Optional[Union[str, Type]] = None,
        num_node_types: int = 1,
        num_edge_types: int = 1,
        concatenate: bool = True,  # True in Barbero et al.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.concatenate = concatenate

        self.learner_type = self._get_learner_name(learner_type)

        self.use_node_type = self.learner_type in self.USES_NODE_TYPE
        self.use_edge_type = self.learner_type in self.USES_EDGE_TYPE
        self.use_features = self.learner_type not in self.NO_FEATURES

        if self.concatenate:
            # Edge-level encoding dimension (matches sheaf learner)
            self.edge_dim = self._compute_edge_dim()
        else:
            # Node-level encoding dimension (for additive)
            self.node_dim = self._compute_node_dim()

        self._build_layers()
        self.reset_parameters()

    def _get_learner_name(self, learner_type) -> Optional[str]:
        if learner_type is None:
            return None
        if isinstance(learner_type, type):
            name = getattr(learner_type, "__name__", str(learner_type))
            class_name_map = {
                "TypeConcatSheafLearner": "type_concat",
                "TypeEnsembleSheafLearner": "type_ensemble",
                "EdgeTypeConcatSheafLearner": "edge_type_concat",
                "NodeTypeConcatSheafLearner": "node_type_concat",
                "TypesOnlySheafLearner": "types_only",
                "NodeTypeSheafLearner": "node_type",
                "EdgeTypeSheafLearner": "edge_type",
                "AttentionTypeConcatSheafLearner": "type_concat",
                "AttentionTypeEnsembleSheafLearner": "type_ensemble",
                "AttentionEdgeEncodingSheafLearner": "edge_type_concat",
                "AttentionNodeEncodingSheafLearner": "node_type_concat",
                "AttentionTypesOnlySheafLearner": "types_only",
                "AttentionNodeTypeSheafLearner": "node_type",
                "AttentionEdgeTypeSheafLearner": "edge_type",
            }
            if name.startswith("Attention"):
                base_name = name[len("Attention") :]
                if base_name in class_name_map:
                    return class_name_map[base_name]
            return class_name_map.get(name, name.lower())

        name = str(learner_type)
        if name.startswith("attention_"):
            name = name[len("attention_") :]
        return name

    # =========================================================================
    # ADDITIVE MODE DIMENSIONS
    # =========================================================================
    def _compute_node_dim(self) -> int:
        """Compute node encoding dimension for additive mode."""
        if not self.use_features and not self.use_node_type:
            return self.in_channels

        dim = 0
        if self.use_features:
            dim += self.in_channels
        if self.use_node_type:
            dim += self.num_node_types

        return dim if dim > 0 else self.in_channels

    # =========================================================================
    # CONCATENATION MODE DIMENSIONS (matches sheaf learner input)
    # =========================================================================
    def _compute_edge_dim(self) -> int:
        """Compute edge encoding dimension for concatenation mode."""
        lt = self.learner_type

        if lt == "type_concat":
            # [x_src || x_dst || t_src || t_dst || t_edge]
            return self.in_channels * 2 + self.num_node_types * 2 + self.num_edge_types

        elif lt == "type_ensemble":
            # [x_src || x_dst] with per-edge-type attention
            return self.in_channels * 2

        elif lt == "edge_type_concat":
            # [x_src || x_dst || t_edge]
            return self.in_channels * 2 + self.num_edge_types

        elif lt == "node_type_concat":
            # [x_src || x_dst || t_src || t_dst]
            return self.in_channels * 2 + self.num_node_types * 2

        elif lt == "types_only":
            # [t_src || t_dst || t_edge]
            return self.num_node_types * 2 + self.num_edge_types

        elif lt == "node_type":
            # [t_src || t_dst]
            return self.num_node_types * 2

        elif lt == "edge_type":
            # [t_edge]
            return self.num_edge_types

        else:
            # Homogeneous: [x_src || x_dst]
            return self.in_channels * 2

    # =========================================================================
    # BUILD LAYERS
    # =========================================================================
    def _build_layers(self):
        """Build attention layers based on mode."""
        if self.concatenate:
            self._build_layers_concat()
        else:
            self._build_layers_additive()

    def _build_layers_additive(self):
        """Build GAT-style additive attention layers."""
        self.lin = nn.Linear(self.node_dim, self.heads * self.node_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, self.heads, self.node_dim))
        self.att_dst = nn.Parameter(torch.empty(1, self.heads, self.node_dim))

        if self.use_edge_type:
            self.att_edge = nn.Parameter(
                torch.empty(1, self.heads, self.num_edge_types)
            )

    def _build_layers_concat(self):
        """Build concatenation-based attention layers."""
        if self.learner_type == "type_ensemble":
            # Per-edge-type attention (like TypeEnsembleSheafLearner)
            self.att = nn.ModuleList(
                [
                    nn.Linear(self.edge_dim, self.heads, bias=False)
                    for _ in range(self.num_edge_types)
                ]
            )
        else:
            self.att = nn.Linear(self.edge_dim, self.heads, bias=False)

    def reset_parameters(self):
        if self.concatenate:
            self._reset_parameters_concat()
        else:
            self._reset_parameters_additive()

    def _reset_parameters_additive(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.use_edge_type:
            nn.init.xavier_uniform_(self.att_edge)

    def _reset_parameters_concat(self):
        if self.learner_type == "type_ensemble":
            for att in self.att:
                nn.init.xavier_uniform_(att.weight)
        else:
            nn.init.xavier_uniform_(self.att.weight)

    # =========================================================================
    # ENCODING
    # =========================================================================
    def _encode_nodes(
        self,
        x: Tensor,
        node_type: Optional[Tensor] = None,
    ) -> Tensor:
        """Augment node features with type information (for additive mode)."""
        parts = []

        if self.use_features:
            parts.append(x)

        if self.use_node_type:
            if node_type is None:
                raise ValueError(
                    f"learner_type='{self.learner_type}' requires node_type"
                )
            node_type_onehot = F.one_hot(node_type, self.num_node_types).float()
            parts.append(node_type_onehot)

        if not parts:
            return x

        return torch.cat(parts, dim=-1)

    def _encode_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_type: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode edges like sheaf learner (for concatenation mode)."""
        src, dst = edge_index
        lt = self.learner_type

        x_src = x[src]
        x_dst = x[dst]

        if self.use_node_type and node_type is not None:
            node_type_onehot = F.one_hot(node_type, self.num_node_types).float()
            t_src = node_type_onehot[src]
            t_dst = node_type_onehot[dst]

        if self.use_edge_type and edge_type is not None:
            t_edge = F.one_hot(edge_type, self.num_edge_types).float()

        if lt == "type_concat":
            return torch.cat([x_src, x_dst, t_src, t_dst, t_edge], dim=-1)
        elif lt == "type_ensemble":
            return torch.cat([x_src, x_dst], dim=-1)
        elif lt == "edge_type_concat":
            return torch.cat([x_src, x_dst, t_edge], dim=-1)
        elif lt == "node_type_concat":
            return torch.cat([x_src, x_dst, t_src, t_dst], dim=-1)
        elif lt == "types_only":
            return torch.cat([t_src, t_dst, t_edge], dim=-1)
        elif lt == "node_type":
            return torch.cat([t_src, t_dst], dim=-1)
        elif lt == "edge_type":
            return t_edge
        else:
            return torch.cat([x_src, x_dst], dim=-1)

    # =========================================================================
    # FORWARD
    # =========================================================================
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        edge_type: Optional[Tensor] = None,
        node_type: Optional[Tensor] = None,
    ) -> Tensor:
        """Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices
            num_nodes: number of nodes N
            edge_type: (E,) edge type indices
            node_type: (N,) node type indices.

        Returns:
            alpha: (E, heads) attention coefficients
        """
        _, dst = edge_index

        if self.concatenate:
            alpha = self._forward_concat(x, edge_index, node_type, edge_type)
        else:
            alpha = self._forward_additive(x, edge_index, node_type, edge_type)

        # LeakyReLU + softmax + dropout
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = pyg_softmax(alpha, dst, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def _forward_additive(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_type: Optional[Tensor],
        edge_type: Optional[Tensor],
    ) -> Tensor:
        """Additive attention: α_src + α_dst + α_edge."""
        src, dst = edge_index

        # Encode nodes
        x_aug = self._encode_nodes(x, node_type)

        # Project: (N, heads, node_dim)
        x_proj = self.lin(x_aug).view(-1, self.heads, self.node_dim)

        # Additive attention
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # (N, heads)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)  # (N, heads)
        alpha = alpha_src[src] + alpha_dst[dst]  # (E, heads)

        # Add edge type contribution
        if self.use_edge_type:
            if edge_type is None:
                raise ValueError(
                    f"learner_type='{self.learner_type}' requires edge_type"
                )
            edge_type_onehot = F.one_hot(edge_type, self.num_edge_types).float()
            alpha_edge = (edge_type_onehot.unsqueeze(1) * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        return alpha

    def _forward_concat(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_type: Optional[Tensor],
        edge_type: Optional[Tensor],
    ) -> Tensor:
        """Concatenation attention: a^T [x_src || x_dst || types]."""
        # Encode edges (matches sheaf learner)
        edge_features = self._encode_edges(x, edge_index, node_type, edge_type)

        if self.learner_type == "type_ensemble":
            # Per-edge-type attention
            return self._forward_concat_ensemble(edge_features, edge_type)
        else:
            return self.att(edge_features)  # (E, heads)

    def _forward_concat_ensemble(
        self,
        edge_features: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """Per-edge-type attention (matches TypeEnsembleSheafLearner)."""
        num_edges = edge_features.size(0)
        device = edge_features.device
        dtype = edge_features.dtype

        unique_types, counts = torch.unique(edge_type, return_counts=True)
        edge_type_idx = torch.argsort(edge_type)
        edge_type_splits = edge_type_idx.split(counts.tolist())

        results = []
        for etype, split in zip(unique_types, edge_type_splits, strict=False):
            results.append(self.att[etype](edge_features[split]))

        stacked = torch.cat(results, dim=0)
        alpha = torch.empty(num_edges, self.heads, device=device, dtype=dtype)
        alpha[edge_type_idx] = stacked

        return alpha
    
    def regenerate_builder(self, num_nodes: int, edge_index: torch.Tensor):
        ab = getattr(self.encoder, "adjacency_builder", None)
        if ab is None:
            return

        if hasattr(ab, "update_graph"):
            ab.update_graph(num_nodes=num_nodes, edge_index=edge_index)
            return

        if isinstance(ab, SheafAdjacencyBuilder):
            new_ab = SheafAdjacencyBuilder(
                num_nodes=num_nodes,
                edge_index=edge_index,
                d=getattr(self.encoder, "d", 1),
                add_self_loops=True,
                normalised=getattr(self.encoder, "normalised", True),
            )
            new_ab.train(ab.training)
            self.encoder.adjacency_builder = new_ab.to(edge_index.device)

    def __repr__(self) -> str:
        mode = self.learner_type or "homogeneous"
        att_mode = "concat" if self.concatenate else "additive"
        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"heads={self.heads}, "
            f"mode={mode}, "
            f"attention={att_mode})"
        )
