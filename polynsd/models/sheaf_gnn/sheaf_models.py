#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from abc import abstractmethod
from typing import Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Adj, InputNodes, OptTensor
import torch_sparse
from .lib import laplace as lap
from polynsd.utils.linalg import cayley_transform

class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""

    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them
    through a linear layer + activation.
    """

    def __init__(
        self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh", **kwargs
    ):
        super(LocalConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2, int(np.prod(out_shape)), bias=False
        )

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        maps = self.linear1(torch.cat([x_src, x_dst], dim=1))
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "local_concat"


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them
    through a linear layer + activation.
    """

    def __init__(
        self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"
    ):
        super(LocalConcatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(
            hidden_channels * 2, int(np.prod(out_shape)), bias=False
        )
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index

        x_src = torch.index_select(x, dim=0, index=src)  # this is really x_src
        x_dst = torch.index_select(x, dim=0, index=dst)  # this is really x_dst
        x_cat = torch.cat([x_src, x_dst], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "local_concat"


class AttentionSheafLearner(SheafLearner):
    """Attention-based sheaf learner for restriction maps.

    Supports both General Sheaf and Bundle Sheaf diffusion:
    - General Sheaf: out_shape=(d, d) -> full d×d restriction matrices
    - Bundle Sheaf: out_shape=(d,) -> d×d orthogonal restriction maps
    """

    def __init__(self, in_channels, out_shape, sheaf_act="tanh", **kwargs):
        super(AttentionSheafLearner, self).__init__()

        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")

        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]

        self.linear1 = torch.nn.Linear(in_channels * 2, self.d**2, bias=False)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        maps = self.linear1(torch.cat([x_src, x_dst], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        attention_maps = id - torch.softmax(maps, dim=-1)

        # Bundle sheaf: apply Cayley transform
        # General sheaf: return as-is
        if not self.is_general:
            return cayley_transform(attention_maps)
        return attention_maps

    def __str__(self):
        return "attention"


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them
    through a linear layer + activation.
    """

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels * 2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_src = torch.index_select(x, dim=0, index=row)
        x_dst = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_src, x_dst], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(
            weights, index=full_right_idx, dim=0
        )
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(
            edge_index, full_matrix=True
        )

    def __str__(self):
        return "edge_weight"


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them
    through a linear layer + activation.
    """

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        maps = self.map_builder(torch.cat([x_src, x_dst], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])

    def __str__(self):
        return "quadratic"


# =====================================================================================
# HETEROGENEOUS SHEAF LEARNERS
# =====================================================================================
class TypeConcatSheafLearner(SheafLearner):
    """HetSheaf-TE (Type-Encoded) - Equation (4)
    Fu⊴e := MLP(xu∥xv∥eϕ(u)∥eϕ(v)∥eψ(e)).

    Concatenates node features with BOTH source and target node type one-hot encodings
    AND edge type one-hot encoding to learn type-specific sheaf restriction maps.

    For each edge (u, v), the input features are:
    [x_u || x_v || one_hot(ϕ(u)) || one_hot(ϕ(v)) || one_hot(ψ(e))]

    Alias: type_concat
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(TypeConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_node_types * 2 + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        # Handle homogeneous graphs where node_types/edge_types may be None
        if node_types is None:
            # Create default node types (all zeros for single type)
            node_types = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_types is None:
            # Create default edge types (all zeros for single type)
            edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types)

        x_cat = torch.cat(
            [x_src, x_dst, src_type, dst_type, edge_type],
            dim=1,
        )

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "type_concat"


class TypeEnsembleSheafLearner(SheafLearner):
    """HetSheaf-ensemble - Equation (10)
    Fu⊴e := MLPψ(e)(xu∥xv).

    Uses a DIFFERENT MLP for each edge type. Allows the model to learn type-specific
    patterns more effectively, at the cost of increased parameters.

    For each edge (u, v) of type ψ(e), the restriction map is:
    Fu⊴e := MLP_ψ(e)(x_u || x_v)

    Alias: type_ensemble
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(TypeEnsembleSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        # self.linear1 = torch.nn.Linear(
        #     in_channels * 2 + num_node_types * 2 + num_edge_types,
        #     int(np.prod(out_shape)),
        #     bias=False,
        # )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.linear1 = nn.ModuleList(
            [
                nn.Linear(in_channels * 2, int(np.prod(out_shape)), bias=False)
                for _ in range(num_edge_types)
            ]
        )

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def compute_map(self, x_cat: torch.Tensor, edge_type):
        return self.linear1[edge_type](x_cat)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)
        x_cat = torch.cat([x_src, x_dst], dim=1)

        # Handle homogeneous graphs where edge_types may be None
        if edge_types is None:
            edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        # Get unique types and their locations
        unique_types, counts = torch.unique(edge_types, return_counts=True)
        edge_type_idx = torch.argsort(edge_types)
        splits = edge_type_idx.split(counts.tolist())

        # Initialize output buffer
        maps = torch.empty(
            (edge_index.size(1), self.linear1[0].out_features),
            device=x.device,
            dtype=x.dtype,
        )

        # Map each group to its specific MLP based on the type value
        for i, type_val in enumerate(unique_types):
            type_int = type_val.item()
            group_indices = splits[i]
            maps[group_indices] = self.linear1[type_int](x_cat[group_indices])

        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "type_ensemble"


class EdgeTypeConcatSheafLearner(SheafLearner):
    """HetSheaf-EE (Edge Encoding) - Equation (5)
    Fu⊴e := MLP(xu∥xv∥eψ(e)).

    Concatenates node features with edge type one-hot encoding only (no node types).
    More natural formulation when only graph edges are heterogeneous.

    For each edge (u, v), the input features are:
    [x_u || x_v || one_hot(ψ(e))]

    Alias: edge_type_concat
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(EdgeTypeConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        # Handle homogeneous graphs where edge_types may be None
        if edge_types is None:
            edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types)

        x_cat = torch.cat(
            [x_src, x_dst, edge_type],
            dim=1,
        )

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "edge_type_concat"


class NodeTypeConcatSheafLearner(SheafLearner):
    """HetSheaf-NE (Node Encoding) - Equation (6)
    Fu⊴e := MLP(xu∥xv∥eϕ(u)∥eϕ(v)).

    Concatenates node features with both source and target node type one-hot encodings
    (no edge type). More natural formulation when only graph nodes are heterogeneous.

    For each edge (u, v), the input features are:
    [x_u || x_v || one_hot(ϕ(u)) || one_hot(ϕ(v))]

    Alias: node_type_concat
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(NodeTypeConcatSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            in_channels * 2 + num_node_types * 2,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        x_src = torch.index_select(x, dim=0, index=src)
        x_dst = torch.index_select(x, dim=0, index=dst)

        # Handle homogeneous graphs where node_types may be None
        if node_types is None:
            node_types = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)

        x_cat = torch.cat(
            [x_src, x_dst, src_type, dst_type],
            dim=1,
        )

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "node_type_concat"


class TypesOnlySheafLearner(SheafLearner):
    """HetSheaf-types - Equation (7)
    Fu⊴e := MLP(eϕ(u)∥eϕ(v)∥eψ(e)).

    Uses ALL available type information (both node types and edge type) but NO node
    features. Requires substantially fewer parameters than feature-based approaches.

    For each edge (u, v), the input features are:
    [one_hot(ϕ(u)) || one_hot(ϕ(v)) || one_hot(ψ(e))]

    Alias: types_only
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(TypesOnlySheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        # Input: 2*one_hot(node_type) + one_hot(edge_type) - NO node features
        self.linear1 = torch.nn.Linear(
            num_node_types * 2 + num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index

        # Handle homogeneous graphs where node_types/edge_types may be None
        if node_types is None:
            node_types = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_types is None:
            edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        # Use only type information, NO node features
        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types)

        x_cat = torch.cat(
            [src_type, dst_type, edge_type],
            dim=1,
        ).to(torch.float)

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "types_only"


class NodeTypeSheafLearner(SheafLearner):
    """HetSheaf-NT (Node Types Only) - Equation (9)
    Fu⊴e := MLP(eϕ(u)∥eϕ(v)).

    Uses ONLY node type information (no node features, no edge types).
    Most parameter-efficient of the type-only approaches for node-heterogeneous graphs.

    For each edge (u, v), the input features are:
    [one_hot(ϕ(u)) || one_hot(ϕ(v))]

    Alias: node_type
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(NodeTypeSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            num_node_types * 2,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        src, dst = edge_index
        
        # Handle homogeneous graphs where node_types may be None
        if node_types is None:
            node_types = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        node_types_one_hot = F.one_hot(node_types, self.num_node_types)
        src_type = torch.index_select(node_types_one_hot, dim=0, index=src)
        dst_type = torch.index_select(node_types_one_hot, dim=0, index=dst)

        x_cat = torch.cat(
            [src_type, dst_type],
            dim=1,
        ).to(torch.float)

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "node_type"


class EdgeTypeSheafLearner(SheafLearner):
    """HetSheaf-ET (Edge Types Only) - Equation (8)
    Fu⊴e := MLP(eψ(e)).

    Uses ONLY edge type information (no node features, no node types).
    Most parameter-efficient of the type-only approaches for edge-heterogeneous graphs.

    For each edge (u, v), the input features are:
    [one_hot(ψ(e))]

    Alias: edge_type
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act: Literal["id", "tanh", "elu"] = "tanh",
        num_node_types: int = 4,
        num_edge_types: int = 12,
    ):
        super(EdgeTypeSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(
            num_edge_types,
            int(np.prod(out_shape)),
            bias=False,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        # Handle homogeneous graphs where edge_types may be None
        if edge_types is None:
            edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        edge_type = F.one_hot(edge_types, num_classes=self.num_edge_types).to(
            torch.float
        )

        maps = self.linear1(edge_type)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])

    def __str__(self):
        return "edge_type"


# =====================================================================================
# ATTENTION-BASED HETEROGENEOUS SHEAF LEARNERS
# =====================================================================================
# These classes implement attention-based sheaf learning with type information.
# Instead of directly predicting restriction maps, they learn attention distributions
# over the stalk dimensions, producing row-stochastic matrices (each row sums to 1).
# Returns I - softmax(M) where M is the learned attention matrix.


class AttentionTypeConcatSheafLearner(SheafLearner):
    """HetSheaf-Attention-TE: Type-Encoded attention sheaf learner.

    Uses attention mechanism with all type information (node types + edge types).
    Input: [x_u || x_v || one_hot(ϕ(u)) || one_hot(ϕ(v)) || one_hot(ψ(e))]

    Produces row-stochastic attention matrices for each edge.
    Returns I - softmax(M) where M is d×d attention matrix.

    Alias: attention_type_concat
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_node_types: int,
        num_edge_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionTypeConcatSheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Input: 2*node_features + 2*one_hot(node_type) + one_hot(edge_type)
        total_in = 2 * in_channels + 2 * num_node_types + num_edge_types
        out_dim = self.d * self.d

        if use_norm:
            self.lin = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )
        else:
            self.lin = nn.Sequential(
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index

        # Get node features
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)

        # Get node type one-hot encodings
        if node_types is not None:
            node_types_onehot = F.one_hot(
                node_types.long(), num_classes=self.num_node_types
            ).float()
            x_type_src = torch.index_select(node_types_onehot, dim=0, index=row)
            x_type_tgt = torch.index_select(node_types_onehot, dim=0, index=col)
        else:
            x_type_src = torch.zeros(row.size(0), self.num_node_types, device=x.device)
            x_type_tgt = torch.zeros(row.size(0), self.num_node_types, device=x.device)

        # Get edge type one-hot encodings
        if edge_types is not None:
            edge_types_onehot = F.one_hot(
                edge_types.long(), num_classes=self.num_edge_types
            ).float()
        else:
            edge_types_onehot = torch.zeros(
                row.size(0), self.num_edge_types, device=x.device
            )

        # Concatenate all features
        h_cat = torch.cat(
            [x_row, x_col, x_type_src, x_type_tgt, edge_types_onehot], dim=-1
        )
        h_sheaf = self.lin(h_cat).view(-1, self.d, self.d)

        # Apply softmax row-wise to get row-stochastic matrices
        attention_maps = torch.softmax(h_sheaf, dim=-1)

        # Return I - attention (standard attention sheaf formulation)
        identity = torch.eye(
            self.d, device=edge_index.device, dtype=h_sheaf.dtype
        ).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_type_concat"


class AttentionEdgeEncodingSheafLearner(SheafLearner):
    """HetSheaf-Attention-EE: Edge-Encoding attention sheaf learner.

    Uses attention mechanism with edge type information only.
    Input: [x_u || x_v || one_hot(ψ(e))]

    Produces row-stochastic attention matrices for each edge.

    Alias: attention_edge_encoding
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_edge_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionEdgeEncodingSheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_edge_types = num_edge_types

        total_in = 2 * in_channels + num_edge_types
        out_dim = self.d * self.d

        if use_norm:
            self.lin = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )
        else:
            self.lin = nn.Sequential(
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)

        if edge_types is not None:
            edge_types_onehot = F.one_hot(
                edge_types.long(), num_classes=self.num_edge_types
            ).float()
        else:
            edge_types_onehot = torch.zeros(
                row.size(0), self.num_edge_types, device=x.device
            )

        h_cat = torch.cat([x_row, x_col, edge_types_onehot], dim=-1)
        h_sheaf = self.lin(h_cat).view(-1, self.d, self.d)

        attention_maps = torch.softmax(h_sheaf, dim=-1)
        identity = torch.eye(
            self.d, device=edge_index.device, dtype=h_sheaf.dtype
        ).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_edge_encoding"


class AttentionNodeEncodingSheafLearner(SheafLearner):
    """HetSheaf-Attention-NE: Node-Encoding attention sheaf learner.

    Uses attention mechanism with node type information only.
    Input: [x_u || x_v || one_hot(ϕ(u)) || one_hot(ϕ(v))]

    Produces row-stochastic attention matrices for each edge.

    Alias: attention_node_encoding
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_node_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionNodeEncodingSheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_node_types = num_node_types

        total_in = 2 * in_channels + 2 * num_node_types
        out_dim = self.d * self.d

        if use_norm:
            self.lin = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )
        else:
            self.lin = nn.Sequential(
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)

        if node_types is not None:
            node_types_onehot = F.one_hot(
                node_types.long(), num_classes=self.num_node_types
            ).float()
            x_type_src = torch.index_select(node_types_onehot, dim=0, index=row)
            x_type_tgt = torch.index_select(node_types_onehot, dim=0, index=col)
        else:
            x_type_src = torch.zeros(row.size(0), self.num_node_types, device=x.device)
            x_type_tgt = torch.zeros(row.size(0), self.num_node_types, device=x.device)

        h_cat = torch.cat([x_row, x_col, x_type_src, x_type_tgt], dim=-1)
        h_sheaf = self.lin(h_cat).view(-1, self.d, self.d)

        attention_maps = torch.softmax(h_sheaf, dim=-1)
        identity = torch.eye(
            self.d, device=edge_index.device, dtype=h_sheaf.dtype
        ).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_node_encoding"


class AttentionTypesOnlySheafLearner(SheafLearner):
    """HetSheaf-Attention-types: Types-only attention sheaf learner.

    Uses attention mechanism with all type information but NO node features.
    Input: [one_hot(ϕ(u)) || one_hot(ϕ(v)) || one_hot(ψ(e))]

    Produces row-stochastic attention matrices for each edge.

    Alias: attention_types_only
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_node_types: int,
        num_edge_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionTypesOnlySheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.hidden_channels = hidden_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        total_in = 2 * num_node_types + num_edge_types
        out_dim = self.d * self.d

        if use_norm:
            self.lin = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )
        else:
            self.lin = nn.Sequential(
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index
        device = edge_index.device

        if node_types is not None:
            node_types_onehot = F.one_hot(
                node_types.long(), num_classes=self.num_node_types
            ).float()
            x_type_src = torch.index_select(node_types_onehot, dim=0, index=row)
            x_type_tgt = torch.index_select(node_types_onehot, dim=0, index=col)
        else:
            x_type_src = torch.zeros(row.size(0), self.num_node_types, device=device)
            x_type_tgt = torch.zeros(row.size(0), self.num_node_types, device=device)

        if edge_types is not None:
            edge_types_onehot = F.one_hot(
                edge_types.long(), num_classes=self.num_edge_types
            ).float()
        else:
            edge_types_onehot = torch.zeros(
                row.size(0), self.num_edge_types, device=device
            )

        h_cat = torch.cat([x_type_src, x_type_tgt, edge_types_onehot], dim=-1)
        h_sheaf = self.lin(h_cat).view(-1, self.d, self.d)

        attention_maps = torch.softmax(h_sheaf, dim=-1)
        identity = torch.eye(self.d, device=device, dtype=h_sheaf.dtype).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_types_only"


class AttentionTypeEnsembleSheafLearner(SheafLearner):
    """HetSheaf-Attention-ensemble: Ensemble attention sheaf learner.

    Uses separate attention MLPs for each edge type.
    For each edge type ψ(e): MLP_ψ(e)(x_u || x_v) → attention matrix

    Produces row-stochastic attention matrices for each edge.

    Alias: attention_type_ensemble
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_edge_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionTypeEnsembleSheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_edge_types = num_edge_types

        out_dim = self.d * self.d

        # Create separate MLPs for each edge type
        self.type_layers = nn.ModuleList()
        for _ in range(num_edge_types):
            if use_norm:
                layer = nn.Sequential(
                    nn.LayerNorm(in_channels * 2),
                    nn.Linear(in_channels * 2, hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channels, out_dim),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(in_channels * 2, hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channels, out_dim),
                )
            self.type_layers.append(layer)

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        h_cat = torch.cat([x_row, x_col], dim=-1)

        if edge_types is None:
            # Use first MLP if no edge types provided
            h_sheaf = self.type_layers[0](h_cat)
        else:
            # Process each edge type separately
            edge_types_long = edge_types.long()
            unique_types, counts = torch.unique(edge_types_long, return_counts=True)

            type_sorted_idx = torch.argsort(edge_types_long)
            type_splits = type_sorted_idx.split(counts.tolist())

            results = []
            for edge_type, split in zip(
                unique_types.tolist(), type_splits, strict=False
            ):
                type_input = h_cat[split]
                type_output = self.type_layers[edge_type](type_input)
                results.append(type_output)

            stacked_maps = torch.cat(results, dim=0)
            h_sheaf = torch.empty_like(stacked_maps)
            h_sheaf[type_sorted_idx] = stacked_maps

        h_sheaf = h_sheaf.view(-1, self.d, self.d)

        attention_maps = torch.softmax(h_sheaf, dim=-1)
        identity = torch.eye(
            self.d, device=edge_index.device, dtype=h_sheaf.dtype
        ).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_type_ensemble"


class AttentionNodeTypeSheafLearner(SheafLearner):
    """HetSheaf-Attention-NT: Node-type-only attention sheaf learner.

    Uses attention mechanism with node type information ONLY (no features, no edge
    types).
    Input: [one_hot(ϕ(u)) || one_hot(ϕ(v))]

    Most parameter-efficient attention learner for node-heterogeneous graphs.

    Alias: attention_node_type
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_node_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionNodeTypeSheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.hidden_channels = hidden_channels
        self.num_node_types = num_node_types

        total_in = 2 * num_node_types
        out_dim = self.d * self.d

        if use_norm:
            self.lin = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )
        else:
            self.lin = nn.Sequential(
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index
        device = edge_index.device

        if node_types is not None:
            node_types_onehot = F.one_hot(
                node_types.long(), num_classes=self.num_node_types
            ).float()
            x_type_src = torch.index_select(node_types_onehot, dim=0, index=row)
            x_type_tgt = torch.index_select(node_types_onehot, dim=0, index=col)
        else:
            x_type_src = torch.zeros(row.size(0), self.num_node_types, device=device)
            x_type_tgt = torch.zeros(row.size(0), self.num_node_types, device=device)

        h_cat = torch.cat([x_type_src, x_type_tgt], dim=-1)
        h_sheaf = self.lin(h_cat).view(-1, self.d, self.d)

        attention_maps = torch.softmax(h_sheaf, dim=-1)
        identity = torch.eye(self.d, device=device, dtype=h_sheaf.dtype).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_node_type"


class AttentionEdgeTypeSheafLearner(SheafLearner):
    """HetSheaf-Attention-ET: Edge-type-only attention sheaf learner.

    Uses attention mechanism with edge type information ONLY (no features, no node
    types).
    Input: [one_hot(ψ(e))]

    Most parameter-efficient attention learner for edge-heterogeneous graphs.

    Alias: attention_edge_type
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        num_edge_types: int,
        hidden_channels: int = 64,
        sheaf_act: str = "tanh",
        use_norm: bool = True,
        **kwargs,
    ):
        super(AttentionEdgeTypeSheafLearner, self).__init__()
        if not isinstance(out_shape, tuple) or len(out_shape) not in [1, 2]:
            raise ValueError(f"out_shape must be 1D or 2D tuple, got {out_shape}")
        self.out_shape = out_shape
        self.is_general = len(out_shape) == 2
        self.d = out_shape[0]
        self.hidden_channels = hidden_channels
        self.num_edge_types = num_edge_types

        total_in = num_edge_types
        out_dim = self.d * self.d

        if use_norm:
            self.lin = nn.Sequential(
                nn.LayerNorm(total_in),
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )
        else:
            self.lin = nn.Sequential(
                nn.Linear(total_in, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, out_dim),
            )

    def forward(
        self,
        x: InputNodes,
        edge_index: Adj,
        edge_types: OptTensor = None,
        node_types: OptTensor = None,
    ):
        row, col = edge_index
        device = edge_index.device

        if edge_types is not None:
            edge_types_onehot = F.one_hot(
                edge_types.long(), num_classes=self.num_edge_types
            ).float()
        else:
            edge_types_onehot = torch.zeros(
                row.size(0), self.num_edge_types, device=device
            )

        h_sheaf = self.lin(edge_types_onehot).view(-1, self.d, self.d)

        attention_maps = torch.softmax(h_sheaf, dim=-1)
        identity = torch.eye(self.d, device=device, dtype=h_sheaf.dtype).unsqueeze(0)
        sheaf_maps = identity - attention_maps

        # Bundle sheaf: apply Cayley transform
        if not self.is_general:
            return cayley_transform(sheaf_maps)
        return sheaf_maps

    def __str__(self):
        return "attention_edge_type"


# from original impl.
class RotationInvariantSheafLearner(SheafLearner):
    def __init__(
        self,
        d: int,
        hidden_channels: int,
        edge_index,
        graph_size,
        out_shape: Tuple[int, ...],
        time_dep: bool,
        transform=None,
        sheaf_act="tanh",
    ):
        super(RotationInvariantSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(d * d, int(np.prod(out_shape)), bias=True)

        self.time_dep = time_dep
        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        right_left_idx = torch.cat(
            (
                self.left_right_idx[1].reshape(1, -1),
                self.left_right_idx[0].reshape(1, -1),
            ),
            0,
        )
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx, right_left_idx), 1)
        self.graph_size = graph_size
        self.dual_laplacian_builder = lap.GeneralLaplacianBuilder(
            edge_index.shape[1],
            sheaf_edge_index_unsorted,
            d=self.d,
            normalised=False,
            deg_normalised=True,
            augmented=False,
        )

        self.transform = transform

        if sheaf_act == "id":
            self.act = lambda x: x
        elif sheaf_act == "tanh":
            self.act = torch.tanh
        elif sheaf_act == "elu":
            self.act = F.elu

    def forward(self, x, edge_index, Maps, **kwargs):
        if Maps is None or not self.time_dep:
            Maps2 = (
                torch.eye(self.d)
                .reshape(1, self.d, self.d)
                .repeat(edge_index.shape[1], 1, 1)
                .to(x.device)
            )
        elif self.transform is not None:
            if self.transform == torch.diag:
                Maps2 = torch.stack([self.transform(map1) for map1 in Maps])
            else:
                Maps2 = self.transform(Maps)
        else:
            Maps2 = Maps
        xT = torch.index_select(
            torch.transpose(
                x.reshape(self.graph_size, -1, self.hidden_channels)[:, 0 : self.d, :],
                -2,
                -1,
            ),
            dim=0,
            index=edge_index[0],
        )
        OldMaps = torch.transpose(Maps2, -1, -2).reshape(-1, self.d)

        xTmaps = torch.cat(
            (
                torch.index_select(xT, 0, self.left_right_idx[0]),
                torch.index_select(xT, 0, self.left_right_idx[1]),
            ),
            0,
        )
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        node_edge_sims = 2 * torch.transpose(
            torch_sparse.spmm(
                Lsheaf[0], Lsheaf[1], OldMaps.size(0), OldMaps.size(0), OldMaps
            ).reshape((-1, self.d, self.d)),
            -2,
            -1,
        )

        node_edge_sims = self.linear1(node_edge_sims.reshape(-1, self.d * self.d))
        maps = self.act(node_edge_sims)
        if len(self.out_shape) == 2:
            maps = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = maps.view(-1, self.out_shape[0])
        return maps
