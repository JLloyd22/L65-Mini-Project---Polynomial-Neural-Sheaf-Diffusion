# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np
import torch_sparse

from typing import Tuple, Optional
from abc import abstractmethod
from torch import nn
from lib import laplace as lap
from models import laplacian_builders as lb


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""

    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index, **kwargs):
        """kwargs may contain:
           - edge_attr: (E, edge_feat_dim) physical edge features e.g. G_ij, B_ij
           - pos: (N, n) node coordinates
           - node_scalars: (N, S) gauge-invariant scalars
           - maps, diff_strength, ... (kept for opinion dynamics variants)
        """
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """
    Learns per-edge sheaf maps by concatenating the features of both endpoint
    nodes and (optionally) the physical edge features, then passing through a
    linear layer and activation.

    When edge_attr is provided (e.g. conductance G_ij and susceptance B_ij),
    the input to the linear layer becomes:
        [x_i || x_j || edge_attr]   shape: [E, 2*in_channels + edge_feat_dim]

    When edge_attr is None the behaviour is identical to the original:
        [x_i || x_j]               shape: [E, 2*in_channels]

    The edge_feat_dim is inferred at construction time so the linear layer is
    sized correctly. Pass edge_feat_dim=0 (default) to disable edge features.
    """

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...],
                 sheaf_act: str = "tanh", edge_feat_dim: int = 0):
        super(LocalConcatSheafLearner, self).__init__()

        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.edge_feat_dim = edge_feat_dim

        # Input size grows by edge_feat_dim when edge features are used
        in_size = in_channels * 2 + edge_feat_dim
        self.linear1 = torch.nn.Linear(in_size, int(np.prod(out_shape)), bias=False)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index, edge_attr: Optional[torch.Tensor] = None, **kwargs):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)

        # Concatenate node features, and edge features if available
        if edge_attr is not None and self.edge_feat_dim > 0:
            x_cat = torch.cat([x_row, x_col, edge_attr], dim=1)
        else:
            x_cat = torch.cat([x_row, x_col], dim=1)

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Variant that can handle an additional dimension and internal reshaping."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...],
                 sheaf_act: str = "tanh", edge_feat_dim: int = 0):
        super(LocalConcatSheafLearnerVariant, self).__init__()

        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.edge_feat_dim = edge_feat_dim
        self.linear1 = torch.nn.Linear(hidden_channels * 2, int(np.prod(out_shape)), bias=False)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index, edge_attr: Optional[torch.Tensor] = None, **kwargs):
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels * 2).sum(dim=1)

        x_cat = self.linear1(x_cat)
        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class AttentionSheafLearner(SheafLearner):
    """
    Sheaf Learner that uses attention to compute the sheaf maps.
    For each edge, learns a d x d matrix using concatenated node features.
    """

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels * 2, d ** 2, bias=False)

    def forward(self, x, edge_index, **kwargs):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """
    For each edge, learns a scalar edge weight from concatenated node features
    via a linear layer and sigmoid.
    """

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()

        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels * 2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index, **kwargs):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)


class QuadraticFormSheafLearner(SheafLearner):
    """
    Learns m = prod(out_shape) quadratic forms on concatenated endpoint features.
    z_e = [x_u || x_v] in R^{2*in_channels}, output q_m(e) = z_e^T M_m z_e.
    """
    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super().__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        D2 = 2 * in_channels
        M0 = torch.eye(D2).unsqueeze(0)
        self.tensor = nn.Parameter(M0.repeat(int(np.prod(out_shape)), 1, 1))

    def forward(self, x, edge_index, **_):
        row, col = edge_index
        x_row = torch.index_select(x, 0, row)
        x_col = torch.index_select(x, 0, col)
        z = torch.cat([x_row, x_col], dim=1)
        q = torch.einsum('ei,mij,ej->em', z, self.tensor, z)
        q = torch.tanh(q)
        if len(self.out_shape) == 2:
            return q.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return q.view(-1, self.out_shape[0])


class RotationInvariantSheafLearner(SheafLearner):
    def __init__(self, d: int, hidden_channels: int, edge_index, graph_size,
                 out_shape: Tuple[int, ...], time_dep: bool, transform=None,
                 sheaf_act: str = "tanh"):
        super(RotationInvariantSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(d * d, int(np.prod(out_shape)), bias=True)

        self.time_dep = time_dep
        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        right_left_idx = torch.cat((self.left_right_idx[1].reshape(1, -1),
                                    self.left_right_idx[0].reshape(1, -1)), 0)
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx, right_left_idx), 1)
        self.graph_size = graph_size
        self.dual_laplacian_builder = lb.GeneralLaplacianBuilder(
            edge_index.shape[1], sheaf_edge_index_unsorted, d=self.d,
            normalised=False, deg_normalised=True, augmented=False)

        self.transform = transform

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu

    def forward(self, x, edge_index, Maps, **kwargs):
        if Maps is None or not self.time_dep:
            Maps2 = torch.eye(self.d).reshape(1, self.d, self.d).repeat(
                edge_index.shape[1], 1, 1).to(x.device)
        elif self.transform is not None:
            if self.transform == torch.diag:
                Maps2 = torch.stack([self.transform(map1) for map1 in Maps])
            else:
                Maps2 = self.transform(Maps)
        else:
            Maps2 = Maps

        xT = torch.index_select(
            torch.transpose(
                x.reshape(self.graph_size, -1, self.hidden_channels)[:, 0:self.d, :], -2, -1),
            dim=0, index=edge_index[0])
        OldMaps = torch.transpose(Maps2, -1, -2).reshape(-1, self.d)

        xTmaps = torch.cat((torch.index_select(xT, 0, self.left_right_idx[0]),
                             torch.index_select(xT, 0, self.left_right_idx[1])), 0)
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        node_edge_sims = 2 * torch.transpose(
            torch_sparse.spmm(Lsheaf[0], Lsheaf[1],
                              OldMaps.size(0), OldMaps.size(0), OldMaps
                              ).reshape((-1, self.d, self.d)), -2, -1)

        node_edge_sims = self.linear1(node_edge_sims.reshape(-1, self.d * self.d))
        maps = self.act(node_edge_sims)
        if len(self.out_shape) == 2:
            maps = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = maps.view(-1, self.out_shape[0])
        return maps