# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from operator import pos
import torch
import torch.nn.functional as F
import torch_sparse
import warnings
from typing import Optional
from torch import nn
from models.sheaf_base import SheafDiffusion
from models.polynomial_sheaf_base import PolynomialSheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant, RotationInvariantSheafLearner
from lib import laplace as lap
from models.orthogonal import Orthogonal


class DiscreteDiagSheafDiffusion(SheafDiffusion):
    '''Discrete Sheaf Diffusion with diagonal-maps.'''

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(
                    self.final_d, self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_attr=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, maps = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                if isinstance(learner, RotationInvariantSheafLearner):
                    maps = learner(x_maps, self.edge_index, maps)
                else:
                    maps = learner(x_maps, self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(maps)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()
            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)


class DiscreteBundleSheafDiffusion(SheafDiffusion):
    '''Discrete Sheaf Diffusion with bundle-sheaf-maps.'''

    def __init__(self, edge_index, args):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)

        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))

        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x, edge_attr=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L, maps = x, None, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                if isinstance(learner, RotationInvariantSheafLearner):
                    maps = learner(x_maps, self.edge_index, maps)
                else:
                    maps = learner(x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.use_act:
                x = F.elu(x)
            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusion(SheafDiffusion):
    '''Discrete Sheaf Diffusion with general sheaf-maps.'''

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        for i in range(self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)

        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def forward(self, x, edge_attr=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L, maps = x, None, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                if isinstance(learner, RotationInvariantSheafLearner):
                    maps = learner(x_maps, self.edge_index, None)
                else:
                    maps = learner(x_maps, self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.use_act:
                x = F.elu(x)
            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        assert torch.all(torch.isfinite(x))
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)


#################################################################################################################################
######################### POLYNOMIAL MODELS #################################################################################
#################################################################################################################################

class DiscreteDiagSheafDiffusionPolynomial(PolynomialSheafDiffusion):
    """Discrete Sheaf Diffusion with diagonal maps + polynomial spectral filter + edge features."""

    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)
        assert args['d'] > 0

        # Edge feature dimension — 0 if no edge features available
        self.edge_feat_dim = int(args.get('edge_feat_dim', 0))

        if self.normalised:
            self.lambda_max = 2.0
        else:
            trivial_maps = torch.ones((edge_index.shape[1], self.d), device=self.device)
            L, _ = lb.DiagLaplacianBuilder(
                self.graph_size, edge_index, d=self.d,
                normalised=self.normalised, deg_normalised=self.deg_normalised,
                add_hp=self.add_hp, add_lp=self.add_lp
            )(trivial_maps)
            (idx_i, idx_j), vals = L
            if self.lambda_max_choice == 'analytic':
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
                print("Analytic bound for lambda_max:", self.lambda_max)
            else:
                N = self.graph_size * self.final_d
                torch.manual_seed(0)
                self.lambda_max = self.estimate_largest_eig((idx_i, idx_j), vals, N)
                print(f"Estimated largest eigenvalue lambda_max: {self.lambda_max}")

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight)
            self.lin_right_weights.append(r)
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight)
            self.lin_left_weights.append(l)

        self.sheaf_learners = nn.ModuleList()
        num_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(self.final_d, self.hidden_channels,
                                                   out_shape=(self.d,), sheaf_act=self.sheaf_act,
                                                   edge_feat_dim=self.edge_feat_dim))
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,),
                                            sheaf_act=self.sheaf_act,
                                            edge_feat_dim=self.edge_feat_dim))

        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons'])
            for _ in range(self.layers)])

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_attr=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, maps = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                xm = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                if isinstance(learner, RotationInvariantSheafLearner):
                    maps = learner(xm, self.edge_index, maps, edge_attr=edge_attr)
                else:
                    maps = learner(xm, self.edge_index, edge_attr=edge_attr)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)
            x_poly = self._poly_eval(idx, vals, x)

            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_act:
                x = F.elu(x)
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x = coeff * x0 - x
            x0 = x

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)


class DiscreteBundleSheafDiffusionPolynomial(PolynomialSheafDiffusion):
    """Discrete Sheaf Diffusion with bundle maps + polynomial spectral filter + edge features."""

    def __init__(self, edge_index, args, K=15):
        super().__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.edge_feat_dim = int(args.get('edge_feat_dim', 0))

        if self.normalised:
            self.lambda_max = 2.0
        else:
            if self.lambda_max_choice == 'analytic':
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
            else:
                E = edge_index.shape[1]
                triv_maps = torch.eye(self.d, device=self.device).unsqueeze(0).expand(E, self.d, self.d)
                tmp_builder = lb.NormConnectionLaplacianBuilder(
                    self.graph_size, edge_index, d=self.d,
                    add_hp=self.add_hp, add_lp=self.add_lp, orth_map=self.orth_trans)
                (idx, vals), _ = tmp_builder(triv_maps)
                Nd = self.graph_size * self.final_d
                self.lambda_max = self.estimate_largest_eig(idx, vals, Nd)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight.data)
            self.lin_right_weights.append(r)
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight.data)
            self.lin_left_weights.append(l)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(self.final_d, self.hidden_channels,
                                                   out_shape=(self.get_param_size(),),
                                                   sheaf_act=self.sheaf_act,
                                                   edge_feat_dim=self.edge_feat_dim))
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(self.hidden_dim,
                                            out_shape=(self.get_param_size(),),
                                            sheaf_act=self.sheaf_act,
                                            edge_feat_dim=self.edge_feat_dim))
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))

        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons'])
            for _ in range(self.layers)])
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

    def get_param_size(self):
        if self.orth_trans in ('matrix_exp', 'cayley'):
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for w in self.weight_learners:
            w.update_edge_index(edge_index)

    def _prepare_maps_for_builder(self, maps: torch.Tensor) -> torch.Tensor:
        E = self.edge_index.size(1)
        expect_full = self.orth_trans in ('matrix_exp', 'cayley')
        P_full = self.d * (self.d + 1) // 2
        P_skew = self.d * (self.d - 1) // 2
        P_skew1 = P_skew + 1

        if maps.dim() == 1:
            maps = maps.unsqueeze(1)
        maps = maps.contiguous().view(E, -1)

        if expect_full:
            if maps.size(1) == P_full:
                return maps
            if maps.size(1) == P_skew1:
                skew = maps[:, :P_skew]
                diag_scalar = maps[:, -1:].expand(-1, self.d)
                return torch.cat([skew, diag_scalar], dim=1)
            raise RuntimeError(f"Expected {P_full} params/edge for '{self.orth_trans}', got {maps.size(1)}")
        else:
            if maps.size(1) == P_skew:
                return maps
            if maps.size(1) == P_full:
                return maps[:, :P_skew]
            raise RuntimeError(f"Expected {P_skew} params/edge for '{self.orth_trans}', got {maps.size(1)}")

    def forward(self, x, edge_attr=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                xm = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                if isinstance(learner, RotationInvariantSheafLearner):
                    maps = learner(xm, self.edge_index, None)
                else:
                    maps = learner(xm, self.edge_index, edge_attr=edge_attr)
                maps = self._prepare_maps_for_builder(maps)

                E = self.edge_index.size(1)
                if self.use_edge_weights:
                    ew = self.weight_learners[layer](xm, self.edge_index)
                    if ew.dim() == 1:
                        ew = ew.unsqueeze(1)
                else:
                    ew = xm.new_ones(E, 1)

                L, trans_maps = self.laplacian_builder(maps, ew)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)
            x_poly = self._poly_eval(idx, vals, x)

            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            if self.use_act:
                x = F.elu(x)
            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusionPolynomial(PolynomialSheafDiffusion):
    """Discrete Sheaf Diffusion with general maps + polynomial spectral filter + edge features."""

    def __init__(self, edge_index, args, K=15):
        super().__init__(edge_index, args)
        assert args['d'] > 1

        self.edge_feat_dim = int(args.get('edge_feat_dim', 0))

        self.polynomial_type = str(args.get('polynomial_type', 'ChebyshevType1'))
        if self.polynomial_type.lower() == 'chebyshev':
            self.polynomial_type = 'ChebyshevType1'
        self.K = int(args.get('poly_layers_K', args.get('chebyshev_layers_K', K)))
        self.gc_lambda = float(args.get('gegenbauer_lambda', 1.0))
        self.jac_alpha = float(args.get('jacobi_alpha', 0.0))
        self.jac_beta = float(args.get('jacobi_beta', 0.0))
        self._eps = 1e-8
        self.lambda_max_choice = args.get('lambda_max_choice', 'analytic')
        assert self.lambda_max_choice in ('analytic', 'iterative', None)

        if self.normalised:
            self.lambda_max = 2.0
        else:
            if self.lambda_max_choice == 'analytic':
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
            else:
                E = edge_index.shape[1]
                triv_maps = torch.eye(self.d, device=self.device).unsqueeze(0).expand(E, self.d, self.d)
                tmp_builder = lb.GeneralLaplacianBuilder(
                    self.graph_size, edge_index, d=self.d,
                    add_lp=self.add_lp, add_hp=self.add_hp,
                    normalised=self.normalised, deg_normalised=self.deg_normalised)
                (idx, vals), _ = tmp_builder(triv_maps)
                Nd = self.graph_size * self.final_d
                self.lambda_max = self.estimate_largest_eig(idx, vals, Nd)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight.data)
            self.lin_right_weights.append(r)
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight.data)
            self.lin_left_weights.append(l)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(self.final_d, self.hidden_channels,
                                                   out_shape=(self.d, self.d),
                                                   sheaf_act=self.sheaf_act,
                                                   edge_feat_dim=self.edge_feat_dim))
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(self.hidden_dim,
                                            out_shape=(self.d, self.d),
                                            sheaf_act=self.sheaf_act,
                                            edge_feat_dim=self.edge_feat_dim))

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons'])
            for _ in range(self.layers)])
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

        if self.polynomial_type == 'Gegenbauer' and not (self.gc_lambda > 0.0):
            warnings.warn("gegenbauer_lambda must be > 0; clamping to 0.1")
            self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == 'Jacobi' and not (self.jac_alpha > -1.0 and self.jac_beta > -1.0):
            warnings.warn("Jacobi requires alpha,beta > -1; clamping to -0.9")
            self.jac_alpha = max(self.jac_alpha, -0.9)
            self.jac_beta = max(self.jac_beta, -0.9)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def forward(self, x, edge_attr=None):
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                xm = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                if isinstance(learner, RotationInvariantSheafLearner):
                    maps = learner(xm, self.edge_index, None)
                else:
                    maps = learner(xm, self.edge_index, edge_attr=edge_attr)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)
            x_poly = self._poly_eval(idx, vals, x)

            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            if self.use_act:
                x = F.elu(x)
            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)
    
    
class PolySpectralGNN(PolynomialSheafDiffusion):
    """
    Polynomial Spectral GNN baseline (non-sheaf).

    Applies the same Chebyshev polynomial filtering as DiscreteDiagSheafDiffusionPolynomial
    but operates on the standard symmetrically-normalised graph Laplacian:
    L_sym = I - D^{-1/2} A D^{-1/2}
    rather than a learned sheaf Laplacian. There are no stalks, no restriction maps,
    and no sheaf learner — d is forced to 1 internally.
    """


    def __init__(self, edge_index, args):
        # Force d=1: PolySpectralGNN is a scalar (non-stalk) model
        args = dict(args)
        args['d'] = 1
        super().__init__(edge_index, args)

        # Standard normalised graph Laplacian has spectrum in [0, 2]
        self.lambda_max = 2.0

        # Build the normalised graph Laplacian once at init (fixed, not learned)
        # L_sym = I - D^{-1/2} A D^{-1/2}
        # stored as sparse (idx, vals) for use in _apply_L / _apply_Lhat
        self._build_normalised_laplacian(edge_index)

        # Linear layers — same structure as DiagSheafPolynomial but d=1
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight)
            self.lin_right_weights.append(r)
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight)
            self.lin_left_weights.append(l)

        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons'])
            for _ in range(self.layers)])

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

    def _build_normalised_laplacian(self, edge_index):
        """
        Builds L_sym = I - D^{-1/2} A D^{-1/2} as a sparse matrix stored
        as (idx, vals) in the same format used by _apply_L.

        For an undirected graph with N nodes and E directed edges:
        - off-diagonal (i,j): -1 / sqrt(deg_i * deg_j)
        - diagonal (i,i):      1.0
        """
        N = self.graph_size
        device = self.device

        row = edge_index[0]
        col = edge_index[1]
        E = row.size(0)

        # Compute degree of each node
        deg = torch.zeros(N, device=device)
        deg.scatter_add_(0, row, torch.ones(E, device=device))

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # Off-diagonal values: -1/sqrt(d_i * d_j)
        off_diag_vals = -deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Diagonal indices and values
        diag_idx = torch.arange(N, device=device)
        diag_vals = torch.ones(N, device=device)

        # Concatenate diagonal and off-diagonal entries
        all_row = torch.cat([diag_idx, row])
        all_col = torch.cat([diag_idx, col])
        all_vals = torch.cat([diag_vals, off_diag_vals])

        # Store as (idx, vals) matching the format expected by _apply_L / torch_sparse.spmm
        self._L_idx = torch.stack([all_row, all_col], dim=0)
        self._L_vals = all_vals

    def forward(self, x, edge_attr=None):
        # edge_attr is ignored — PolySpectralGNN uses no edge features
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout,
                      training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        # d=1 so graph_size * final_d = graph_size
        x = x.view(self.graph_size * self.final_d, -1)

        idx = self._L_idx
        vals = self._L_vals
        x0 = x

        for layer in range(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()
            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            # Polynomial filter on standard graph Laplacian
            Lx0 = self._apply_L(idx, vals, x)
            x_poly = self._poly_eval(idx, vals, x)

            # High-pass correction
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp

            if self.use_act:
                x = F.elu(x)

            # Gated residual
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x = coeff * x0 - x
            x0 = x

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)