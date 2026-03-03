#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor, nn
from torch_geometric.data import Data
import warnings
from .. import laplacian_builders as lb
from polynsd.models.sheaf_gnn import sheaf_attention as san
from polynsd.models.sheaf_gnn.orthogonal import Orthogonal
from polynsd.models.sheaf_gnn.sheaf_base import SheafDiffusion
from polynsd.models.sheaf_gnn.config import SheafLearners
from polynsd.models.sheaf_gnn.sheaf_models import (
    EdgeWeightLearner,
    LocalConcatSheafLearnerVariant,
    RotationInvariantSheafLearner,
    cayley_transform,
)
from polynsd.utils.linalg import estimate_largest_eig
from ..utils import init_sheaf_learner

MATRIX_SHEAF_LEARNERS = {
    SheafLearners.attention,
    SheafLearners.attention_type_concat,
    SheafLearners.attention_type_ensemble,
    SheafLearners.attention_edge_encoding,
    SheafLearners.attention_node_encoding,
    SheafLearners.attention_types_only,
    SheafLearners.attention_node_type,
    SheafLearners.attention_edge_type,
}


class DiscreteSheafDiffusion(SheafDiffusion):
    def __init__(
        self,
        edge_index,
        args,
        sheaf_learner: str = "local_concat",
    ):
        super(DiscreteSheafDiffusion, self).__init__(edge_index, args)
        self.sheaf_type = (
            sheaf_learner
            if isinstance(sheaf_learner, SheafLearners)
            else SheafLearners(sheaf_learner)
        )
        self.sheaf_learner = init_sheaf_learner(self.sheaf_type)
        print("Using sheaf learner:", self.sheaf_type, flush=True)

    @abstractmethod
    def process_restriction_maps(self, maps): ...

    @abstractmethod
    def forward(self, data: Data): ...

    def regenerate_builder(self, num_nodes: int, edge_index: torch.Tensor):
        ''' 
            Rebuild or update the laplacian builder if it exists in the model.
        '''
        lb = getattr(self, "laplacian_builder", None)
        if lb is None:
            return

        if hasattr(lb, "update_graph"):
            size_update = num_nodes if getattr(lb, "size", num_nodes) != num_nodes else None
            lb.update_graph(edge_index=edge_index, size=size_update)
            return

        if hasattr(lb, "create_with_new_edge_index"):
            if getattr(lb, "size", num_nodes) != num_nodes:
                lb.size = num_nodes
            new_lb = lb.create_with_new_edge_index(edge_index)
            new_lb.train(lb.training)
            self.laplacian_builder = new_lb.to(edge_index.device)

class DiscreteDiagSheafDiffusion(DiscreteSheafDiffusion):
    def __init__(self, edge_index, args, sheaf_learner):
        super(DiscreteDiagSheafDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )
        assert args.d > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for _i in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                )
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for _i in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        self.epsilons = nn.ParameterList()
        for _ in range(self.layers):
            self.epsilons.append(
                nn.Parameter(
                    torch.zeros((self.final_d, 1)),
                    requires_grad=getattr(args, "use_epsilons", True),
                )
            )

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data: Data):
        if self.use_embedding:
            x = F.dropout(data.x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)
        else:
            x = data.x
        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )

                # maps are the linear restriction maps
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.graph_size, -1),
                    self.edge_index,
                    data.edge_type,
                    data.node_type,
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

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

            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        return x, maps

    def process_restriction_maps(self, maps):
        return maps

    def __str__(self):
        return f"DiagSheaf-{self.sheaf_type}"


class DiscreteBundleSheafDiffusion(DiscreteSheafDiffusion):
    def __init__(self, edge_index, args, sheaf_learner):
        super(DiscreteBundleSheafDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )
        assert args.d > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for _i in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                )
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for _i in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                out_shape = (
                    (self.d, self.d)
                    if self.sheaf_type in MATRIX_SHEAF_LEARNERS
                    else (self.get_param_size(),)
                )
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=out_shape,
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )

            if self.use_edge_weights:
                self.weight_learners.append(
                    EdgeWeightLearner(self.hidden_dim, edge_index)
                )
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(
                nn.Parameter(
                    torch.zeros((self.final_d, 1)),
                    requires_grad=getattr(args, "use_epsilons", True),
                )
            )

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ["matrix_exp", "cayley"]:
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

    def forward(self, data: Data):
        if self.use_embedding:
            x = F.dropout(data.x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)
        else:
            x = data.x
        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](
                    x_maps, self.edge_index, data.edge_type, data.node_type
                )
                edge_weights = (
                    self.weight_learners[layer](x_maps, self.edge_index)
                    if self.use_edge_weights
                    else None
                )
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (
                1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
            ) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        return x, maps

    def process_restriction_maps(self, maps: torch.Tensor) -> torch.Tensor:
        transform = Orthogonal(self.d, self.orth_trans)
        maps = transform(maps)
        return torch.flatten(maps, start_dim=1, end_dim=-1)

    def __str__(self):
        return f"BundleSheaf-{self.sheaf_type}"


class DiscreteGeneralSheafDiffusion(DiscreteSheafDiffusion):
    def __init__(self, edge_index, args, sheaf_learner):
        super(DiscreteGeneralSheafDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )
        assert args.d > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for _i in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                )
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for _i in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_lp=self.add_lp,
            add_hp=self.add_hp,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
        )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(
                nn.Parameter(
                    torch.zeros((self.final_d, 1)),
                    requires_grad=getattr(args, "use_epsilons", True),
                )
            )

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

    def forward(self, data: Data):
        if self.use_embedding:
            x = F.dropout(data.x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)
        else:
            x = data.x
        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[layer](
                    x_maps.reshape(self.graph_size, -1),
                    self.edge_index,
                    data.edge_type,
                    data.node_type,
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (
                1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
            ) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        return x, maps

    def process_restriction_maps(self, maps):
        return torch.flatten(maps, start_dim=1, end_dim=-1)

    def __str__(self):
        return f"GeneralSheaf-{self.sheaf_type}"


#######################################################
# POLYNOMIAL MODELS ( Orthogonal bases on [−1,1] ) 
# #####################################################


class DiscreteDiagSheafDiffusionPolynomial(DiscreteSheafDiffusion):
    """Discrete Sheaf Diffusion with diagonal maps + configurable polynomial spectral filter.

    polynomial_type ∈ {
        'Chebyshev', 'ChebyshevType1', 'ChebyshevType2', 'ChebyshevType3', 'ChebyshevType4',
        'Legendre', 'Gegenbauer', 'Jacobi'
    }

    - Chebyshev / ChebyshevType1: T_k (1st kind)
    - ChebyshevType2: U_k  (2nd kind)
    - ChebyshevType3: V_k  (3rd kind)
    - ChebyshevType4: W_k  (4th kind)
    - Legendre:       P_k
    - Gegenbauer:     C_k^{(λ)} with λ>0 (args['gegenbauer_lambda'])
    - Jacobi:         P_k^{(α,β)} with α,β>-1 (args['jacobi_alpha'], args['jacobi_beta'])
    """

    def __init__(self, edge_index, args, sheaf_learner, K=15):
        super().__init__(edge_index, args, sheaf_learner)
        assert args.d > 0

        # ---- Polynomial Configuration ----
        self.polynomial_type = str(getattr(args, "polynomial_type", "ChebyshevType1"))
        # Treating 'Chebyshev'(with no indication w.r.t. the type) as alias for first kind (T_k).
        if self.polynomial_type.lower() == "chebyshev":
            self.polynomial_type = "ChebyshevType1"

        # Order K.
        self.K = int(
            getattr(args, "poly_layers_K", getattr(args, "chebyshev_layers_K", K))
        )

        # Parameters for each of the families.
        self.gc_lambda = float(getattr(args, "gegenbauer_lambda", 1.0))  # > 0
        self.jac_alpha = float(getattr(args, "jacobi_alpha", 0.0))  # > -1
        self.jac_beta = float(getattr(args, "jacobi_beta", 0.0))  # > -1
        self._eps = 1e-8  # Small numeric guard

        # ---- λ_max Handling: Set its value depending on the type of sheaf laplacian we use. ----
        lambda_max_choice = getattr(args, "lambda_max_choice", "analytic")
        if lambda_max_choice is not None:
            assert lambda_max_choice in ("analytic", "iterative")
        self.lambda_max_choice = lambda_max_choice

        # If the Sheaf Laplacian is Normalised, since spectrum is bounded in [0,2], set it =2.
        if self.normalised:
            self.lambda_max = 2.0
        # If the Sheaf Laplacian is not Normalised.
        else:
            # Build initial trivial maps to estimate degrees / power-iterate.
            trivial_maps = torch.ones((edge_index.shape[1], self.d), device=self.device)
            L, _ = lb.DiagLaplacianBuilder(
                self.graph_size,
                edge_index,
                d=self.d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=self.add_hp,
                add_lp=self.add_lp,
            )(trivial_maps)

            (idx_i, idx_j), vals = L
            # If the Choice is analytic, set it using Gershgorin's Theorem.
            if lambda_max_choice == "analytic":
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
                print("Analytic bound for λ_max:", self.lambda_max)
            # If the Choice is Iterative, set it using Rayleight Iteration.
            else:
                N = self.graph_size * self.final_d
                self.lambda_max = estimate_largest_eig((idx_i, idx_j), vals, N)
                print(f"Estimated largest eigenvalue λ_max: {self.lambda_max}")

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        # TODO: DIFF self.batch_norms = nn.ModuleList() però poi non usate
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight)
            self.lin_right_weights.append(r)
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight)
            self.lin_left_weights.append(l)

        # ---- Sheaf Learners (Diag) ----
        self.sheaf_learners = nn.ModuleList()
        num_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        in_channels=self.hidden_dim,
                        out_shape=(self.d,),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )

        # ---- Laplacian Builder (Diag) ----
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        # ---- Residual Epsilons ----
        # self.epsilons = nn.ParameterList([
        #    nn.Parameter(torch.zeros((self.final_d, 1)),
        #                 requires_grad=args['use_epsilons'])
        #    for _ in range(self.layers)
        # ])
        self.epsilons = nn.ParameterList()
        for _ in range(self.layers):
            self.epsilons.append(
                nn.Parameter(
                    torch.zeros((self.final_d, 1)),
                    requires_grad=getattr(args, "use_epsilons", True),
                )
            )

        # ---- Embedding/Projection ----
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # ---- Polynomial Coefficients (Convex combo, like  Chebyshev) ----
        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))
        assert self.poly_logits.numel() == self.K + 1

        # ---- High-Pass Skip for contrasting Oversmoothing Bias ----
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

        # Sanity checks for parameters.
        if self.polynomial_type == "Gegenbauer":
            if not (self.gc_lambda > 0.0):
                warnings.warn(
                    "gegenbauer_lambda must be > 0; clamping to 0.1", stacklevel=2
                )
                self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == "Jacobi":
            if not (self.jac_alpha > -1.0 and self.jac_beta > -1.0):
                warnings.warn(
                    "Jacobi requires alpha,beta > -1; clamping to -0.9", stacklevel=2
                )
                self.jac_alpha = max(self.jac_alpha, -0.9)
                self.jac_beta = max(self.jac_beta, -0.9)

    # ---------- Utilities: L@v and Lhat@v ----------
    def _apply_L(self, idx, vals, v):
        return torch_sparse.spmm(idx, vals, v.size(0), v.size(0), v)

    def _apply_Lhat(self, idx, vals, v):
        # scale L to [-1,1]: Lhat = (2/λ_max) * L - I
        Lv = self._apply_L(idx, vals, v)
        return (2.0 / self.lambda_max) * Lv - v

    # ---------- Polynomial Evaluation on a vector x ----------
    def _poly_eval(self, idx, vals, x):
        """Return p(L) x using the chosen polynomial basis and learned coefficients."""
        K = self.K
        w = F.softmax(self.poly_logits, dim=0)  # Convex mixture.

        # Helpers for consistent accumulation.
        def add(acc, k, vec):
            return acc + (w[k] * vec)

        # We need Lhat for all bases used here.
        def Lhat(v):
            return self._apply_Lhat(idx, vals, v)

        poly = self.polynomial_type

        # --- Chebyshev Family ---
        if poly in ("ChebyshevType1", "Chebyshev"):
            # T0 = x; T1 = Lhat x; T_{k+1} = 2 Lhat T_k - T_{k-1}
            T0 = x
            T1 = Lhat(x) if K >= 1 else None
            out = w[0] * T0
            if K >= 1:
                out = add(out, 1, T1)
            for k in range(1, K):
                LT1 = Lhat(T1)
                Tk1 = 2.0 * LT1 - T0
                out = add(out, k + 1, Tk1)
                T0, T1 = T1, Tk1
            return out

        if poly == "ChebyshevType2":
            # U0 = x; U1 = 2 Lhat x; U_{k+1} = 2 Lhat U_k - U_{k-1}
            U0 = x
            U1 = 2.0 * Lhat(x) if K >= 1 else None
            out = w[0] * U0
            if K >= 1:
                out = add(out, 1, U1)
            for k in range(1, K):
                LU1 = Lhat(U1)
                Uk1 = 2.0 * LU1 - U0
                out = add(out, k + 1, Uk1)
                U0, U1 = U1, Uk1
            return out

        if poly == "ChebyshevType3":
            # V0 = x; V1 = 2 Lhat x - x; V_{k+1} = 2 Lhat V_k - V_{k-1}
            V0 = x
            V1 = (2.0 * Lhat(x) - x) if K >= 1 else None
            out = w[0] * V0
            if K >= 1:
                out = add(out, 1, V1)
            for k in range(1, K):
                LV1 = Lhat(V1)
                Vk1 = 2.0 * LV1 - V0
                out = add(out, k + 1, Vk1)
                V0, V1 = V1, Vk1
            return out

        if poly == "ChebyshevType4":
            # W0 = x; W1 = 2 Lhat x + x; W_{k+1} = 2 Lhat W_k - W_{k-1}
            W0 = x
            W1 = (2.0 * Lhat(x) + x) if K >= 1 else None
            out = w[0] * W0
            if K >= 1:
                out = add(out, 1, W1)
            for k in range(1, K):
                LW1 = Lhat(W1)
                Wk1 = 2.0 * LW1 - W0
                out = add(out, k + 1, Wk1)
                W0, W1 = W1, Wk1
            return out

        # --- Legendre ---
        if poly == "Legendre":
            # P0 = x; P1 = Lhat x
            P0 = x
            P1 = Lhat(x) if K >= 1 else None
            out = w[0] * P0
            if K >= 1:
                out = add(out, 1, P1)
            for k in range(1, K):
                ak = (2.0 * k + 1.0) / (k + 1.0)
                ck = k / (k + 1.0)
                LP1 = Lhat(P1)
                Pk1 = ak * LP1 - ck * P0
                out = add(out, k + 1, Pk1)
                P0, P1 = P1, Pk1
            return out

        # --- Gegenbauer (λ>0) ---
        if poly == "Gegenbauer":
            lam = max(self.gc_lambda, 1e-3)
            C0 = x
            C1 = (2.0 * lam) * Lhat(x) if K >= 1 else None
            out = w[0] * C0
            if K >= 1:
                out = add(out, 1, C1)
            for k in range(1, K):
                ak = 2.0 * (k + lam) / (k + 1.0)
                ck = (k + 2.0 * lam - 1.0) / (k + 1.0)
                LC1 = Lhat(C1)
                Ck1 = ak * LC1 - ck * C0
                out = add(out, k + 1, Ck1)
                C0, C1 = C1, Ck1
            return out

        # --- Jacobi (α,β>-1) ---
        if poly == "Jacobi":
            a = self.jac_alpha
            b = self.jac_beta
            # P0 = x
            P0 = x
            out = w[0] * P0
            if K >= 1:
                # P1 = c1 * Lhat(P0) + c0 * P0
                den = a + b + 2.0
                c1 = den / 2.0
                c0 = (a - b) / (den + 0.0)
                P1 = c1 * Lhat(P0) + c0 * P0
                out = add(out, 1, P1)
                for k in range(1, K):
                    # P_{k+1} = (A_k * Lhat + B_k) P_k - C_k P_{k-1}
                    den1 = 2.0 * k + a + b
                    den2 = den1 + 2.0
                    # A_k
                    Ak = (
                        2.0
                        * (k + 1.0)
                        * (k + a + b + 1.0)
                        / ((den1 + 1.0) * den2 + self._eps)
                    )
                    # B_k
                    Bk = (b * b - a * a) / (den1 * den2 + self._eps)
                    # C_k
                    Ck = 2.0 * (k + a) * (k + b) / (den1 * (den1 + 1.0) + self._eps)

                    LP1 = Lhat(P1)
                    Pk1 = Ak * LP1 + Bk * P1 - Ck * P0
                    out = add(out, k + 1, Pk1)
                    P0, P1 = P1, Pk1
            return out

        raise ValueError(f"Unknown polynomial_type: {self.polynomial_type}")

    def forward(self, data: Data):
        x = data.x
        # 1) Embedding + Dropout + Act.
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        # 2) Optional Second Linear + Sheaf-dropout -> Flatten.
        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        # 3) Diffusion Layers.
        x0, maps = x, None
        for layer in range(self.layers):
            # (Re)learn diag maps and build Laplacian.
            if layer == 0 or self.nonlinear:
                xm = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                maps = (
                    learner(xm, self.edge_index, maps)
                    if isinstance(learner, RotationInvariantSheafLearner)
                    else learner(xm, self.edge_index, data.edge_type, data.node_type)
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            # ---- Polynomial Spectral Filtering p(L) x ----
            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)  # for HP skip
            x_poly = self._poly_eval(idx, vals, x)

            # 4) High-pass Reinjection STep.
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp

            # 5) Residual + Nonlinearity.
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_act:
                x = F.elu(x)
            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
            x = coeff * x0 - x
            x0 = x

        # this part is commented because in Luke implementation there is no final linear layer in this part of the code, it is done in the pl module.
        # 6) Projection to output.
        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        return x, maps

    def process_restriction_maps(self, maps: torch.Tensor) -> torch.Tensor:
        return maps

    def __str__(self):
        return f"DiagSheafPoly-{self.sheaf_type}-{self.polynomial_type}"


class DiscreteBundleSheafDiffusionPolynomial(DiscreteSheafDiffusion):
    """Discrete Sheaf Diffusion with bundle maps + configurable polynomial spectral filter + HP skip."""

    def __init__(self, edge_index, args, sheaf_learner, K=15):
        super().__init__(edge_index, args, sheaf_learner)
        assert args.d > 1
        assert not self.deg_normalised

        # ---- Polynomial config / λ_max ----
        self.polynomial_type = str(getattr(args, "polynomial_type", "ChebyshevType1"))
        if self.polynomial_type.lower() == "chebyshev":
            self.polynomial_type = "ChebyshevType1"

        self.K = int(
            getattr(args, "poly_layers_K", getattr(args, "chebyshev_layers_K", K))
        )
        self.gc_lambda = float(getattr(args, "gegenbauer_lambda", 1.0))  # > 0
        self.jac_alpha = float(getattr(args, "jacobi_alpha", 0.0))  # > -1
        self.jac_beta = float(getattr(args, "jacobi_beta", 0.0))  # > -1
        self._eps = 1e-8

        self.lambda_max_choice = getattr(args, "lambda_max_choice", "analytic")
        assert self.lambda_max_choice in ("analytic", "iterative", None)

        if self.normalised:
            self.lambda_max = 2.0
        else:
            if self.lambda_max_choice == "analytic":
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
            else:
                E = edge_index.shape[1]
                triv_maps = (
                    torch.eye(self.d, device=self.device)
                    .unsqueeze(0)
                    .expand(E, self.d, self.d)
                )
                tmp_builder = lb.NormConnectionLaplacianBuilder(
                    self.graph_size,
                    edge_index,
                    d=self.d,
                    add_hp=self.add_hp,
                    add_lp=self.add_lp,
                    orth_map=self.orth_trans,
                )
                (idx, vals), _ = tmp_builder(triv_maps)
                Nd = self.graph_size * self.final_d
                self.lambda_max = estimate_largest_eig(idx, vals, Nd)

        # ---- Linear Maps ----
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight.data)
            self.lin_right_weights.append(r)
        for _ in range(self.layers):
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight.data)
            self.lin_left_weights.append(l)

        # ---- Sheaf learners / Edge weights ----
        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        self.hidden_dim,
                        out_shape=(self.get_param_size(),),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )
            if self.use_edge_weights:
                self.weight_learners.append(
                    EdgeWeightLearner(self.hidden_dim, edge_index)
                )

        # ---- Laplacian Builder ----
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

        # ---- Residual Epsilons, Embed/Proj, Polynomial Mix ----
        self.epsilons = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros((self.final_d, 1)),
                    requires_grad=getattr(args, "use_epsilons", True),
                )
                for _ in range(self.layers)
            ]
        )
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))  # softmax mixture
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

        if self.polynomial_type == "Gegenbauer" and not (self.gc_lambda > 0.0):
            warnings.warn(
                "gegenbauer_lambda must be > 0; clamping to 0.1", stacklevel=2
            )
            self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == "Jacobi" and not (
            self.jac_alpha > -1.0 and self.jac_beta > -1.0
        ):
            warnings.warn(
                "Jacobi requires alpha,beta > -1; clamping to -0.9", stacklevel=2
            )
            self.jac_alpha = max(self.jac_alpha, -0.9)
            self.jac_beta = max(self.jac_beta, -0.9)

    def get_param_size(self):
        # Match the Builder’s Expectation:
        # - matrix_exp / cayley: allow skew + diag  -> d(d+1)/2
        # - others (e.g., householder): skew only    -> d(d-1)/2
        if self.orth_trans in ("matrix_exp", "cayley"):
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
        """Ensure correct per-edge parameter count for the chosen orth_map."""
        E = self.edge_index.size(1)
        expect_full = self.orth_trans in ("matrix_exp", "cayley")
        P_full = self.d * (self.d + 1) // 2
        P_skew = self.d * (self.d - 1) // 2
        P_skew1 = P_skew + 1  # (Skew + Single diag scalar), optional convenience.

        if maps.dim() == 1:
            maps = maps.unsqueeze(1)
        maps = maps.contiguous().view(E, -1)

        if expect_full:
            if maps.size(1) == P_full:
                return maps
            if maps.size(1) == P_skew1:
                skew = maps[:, :P_skew]
                diag_scalar = maps[:, -1:].expand(-1, self.d)  # (E, d)
                return torch.cat([skew, diag_scalar], dim=1)
            raise RuntimeError(
                f"Expected {P_full} (or {P_skew1}) params/edge for '{self.orth_trans}', got {maps.size(1)}"
            )
        else:
            if maps.size(1) == P_skew:
                return maps
            if maps.size(1) == P_full:
                return maps[:, :P_skew]
            raise RuntimeError(
                f"Expected {P_skew} params/edge for '{self.orth_trans}', got {maps.size(1)}"
            )

    # ---------- L and Lhat ----------
    def _apply_L(self, idx, vals, v):
        return torch_sparse.spmm(idx, vals, v.size(0), v.size(0), v)

    def _apply_Lhat(self, idx, vals, v):
        Lv = self._apply_L(idx, vals, v)
        return (2.0 / self.lambda_max) * Lv - v

    # ---------- Polynomial Evaluation ----------
    def _poly_eval(self, idx, vals, x):
        K = self.K
        w = F.softmax(self.poly_logits, dim=0)

        def Lhat(v):
            return self._apply_Lhat(idx, vals, v)

        def add(acc, k, vec):
            return acc + (w[k] * vec)

        poly = self.polynomial_type

        if poly in ("ChebyshevType1", "Chebyshev"):
            T0 = x
            out = w[0] * T0
            if K >= 1:
                T1 = Lhat(x)
                out = add(out, 1, T1)
                for k in range(1, K):
                    LT1 = Lhat(T1)
                    Tk1 = 2.0 * LT1 - T0
                    out = add(out, k + 1, Tk1)
                    T0, T1 = T1, Tk1
            return out

        if poly == "ChebyshevType2":
            U0 = x
            out = w[0] * U0
            if K >= 1:
                U1 = 2.0 * Lhat(x)
                out = add(out, 1, U1)
                for k in range(1, K):
                    LU1 = Lhat(U1)
                    Uk1 = 2.0 * LU1 - U0
                    out = add(out, k + 1, Uk1)
                    U0, U1 = U1, Uk1
            return out

        if poly == "ChebyshevType3":
            V0 = x
            out = w[0] * V0
            if K >= 1:
                V1 = 2.0 * Lhat(x) - x
                out = add(out, 1, V1)
                for k in range(1, K):
                    LV1 = Lhat(V1)
                    Vk1 = 2.0 * LV1 - V0
                    out = add(out, k + 1, Vk1)
                    V0, V1 = V1, Vk1
            return out

        if poly == "ChebyshevType4":
            W0 = x
            out = w[0] * W0
            if K >= 1:
                W1 = 2.0 * Lhat(x) + x
                out = add(out, 1, W1)
                for k in range(1, K):
                    LW1 = Lhat(W1)
                    Wk1 = 2.0 * LW1 - W0
                    out = add(out, k + 1, Wk1)
                    W0, W1 = W1, Wk1
            return out

        if poly == "Legendre":
            P0 = x
            out = w[0] * P0
            if K >= 1:
                P1 = Lhat(x)
                out = add(out, 1, P1)
                for k in range(1, K):
                    ak = (2.0 * k + 1.0) / (k + 1.0)
                    ck = k / (k + 1.0)
                    LP1 = Lhat(P1)
                    Pk1 = ak * LP1 - ck * P0
                    out = add(out, k + 1, Pk1)
                    P0, P1 = P1, Pk1
            return out

        if poly == "Gegenbauer":
            lam = max(self.gc_lambda, 1e-3)
            C0 = x
            out = w[0] * C0
            if K >= 1:
                C1 = (2.0 * lam) * Lhat(x)
                out = add(out, 1, C1)
                for k in range(1, K):
                    ak = 2.0 * (k + lam) / (k + 1.0)
                    ck = (k + 2.0 * lam - 1.0) / (k + 1.0)
                    LC1 = Lhat(C1)
                    Ck1 = ak * LC1 - ck * C0
                    out = add(out, k + 1, Ck1)
                    C0, C1 = C1, Ck1
            return out

        if poly == "Jacobi":
            a, b = self.jac_alpha, self.jac_beta
            P0 = x
            out = w[0] * P0
            if K >= 1:
                den = a + b + 2.0
                c1 = den / 2.0
                c0 = (a - b) / (den + 0.0)
                P1 = c1 * Lhat(P0) + c0 * P0
                out = add(out, 1, P1)
                for k in range(1, K):
                    den1 = 2.0 * k + a + b
                    den2 = den1 + 2.0
                    Ak = (
                        2.0
                        * (k + 1.0)
                        * (k + a + b + 1.0)
                        / ((den1 + 1.0) * den2 + self._eps)
                    )
                    Bk = (b * b - a * a) / (den1 * den2 + self._eps)
                    Ck = 2.0 * (k + a) * (k + b) / (den1 * (den1 + 1.0) + self._eps)
                    LP1 = Lhat(P1)
                    Pk1 = Ak * LP1 + Bk * P1 - Ck * P0
                    out = add(out, k + 1, Pk1)
                    P0, P1 = P1, Pk1
            return out

        raise ValueError(f"Unknown polynomial_type: {self.polynomial_type}")

    def forward(self, data: Data):
        x = data.x
        # 1) Embedding + Dropout + Act.
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        # 2) Optional Second Linear + Sheaf-dropout -> Flatten.
        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        # 3) Diffusion Layers.
        x0, maps = x, None
        for layer in range(self.layers):
            # (Re)learn maps and build Laplacian.
            if layer == 0 or self.nonlinear:
                xm = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                xm = xm.reshape(self.graph_size, -1)

                learner = self.sheaf_learners[layer]
                maps = (
                    learner(xm, self.edge_index, maps)
                    if isinstance(learner, RotationInvariantSheafLearner)
                    else learner(xm, self.edge_index, data.edge_type, data.node_type)
                )
                maps = self._prepare_maps_for_builder(maps)

                # Edge weights -> (E,1).
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

            # Linear sandwich.
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )

            # Polynomial Filter p(L) x .
            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)  # For HP skip.
            x_poly = self._poly_eval(idx, vals, x)

            # High-pass skip + Activation + Residual.
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            if self.use_act:
                x = F.elu(x)

            x0 = (
                1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
            ) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        return x, maps

    def process_restriction_maps(self, maps: torch.Tensor) -> torch.Tensor:
        transform = Orthogonal(self.d, self.orth_trans)
        maps = transform(maps)
        return torch.flatten(maps, start_dim=1, end_dim=-1)

    def __str__(self):
        return f"BundleSheafPoly-{self.sheaf_type}-{self.polynomial_type}"


class DiscreteGeneralSheafDiffusionPolynomial(DiscreteSheafDiffusion):
    """Discrete Sheaf Diffusion with general maps + configurable polynomial spectral filter + HP skip."""

    def __init__(self, edge_index, args, sheaf_learner, K=15):
        super().__init__(edge_index, args, sheaf_learner)
        assert args.d > 1

        # ---- Polynomial Config / λ_max ----
        self.polynomial_type = str(getattr(args, "polynomial_type", "ChebyshevType1"))
        if self.polynomial_type.lower() == "chebyshev":
            self.polynomial_type = "ChebyshevType1"

        self.K = int(
            getattr(args, "poly_layers_K", getattr(args, "chebyshev_layers_K", K))
        )
        self.gc_lambda = float(getattr(args, "gegenbauer_lambda", 1.0))  # > 0
        self.jac_alpha = float(getattr(args, "jacobi_alpha", 0.0))  # > -1
        self.jac_beta = float(getattr(args, "jacobi_beta", 0.0))  # > -1
        self._eps = 1e-8

        self.lambda_max_choice = getattr(args, "lambda_max_choice", "analytic")
        assert self.lambda_max_choice in ("analytic", "iterative", None)

        if self.normalised:
            self.lambda_max = 2.0
        else:
            if self.lambda_max_choice == "analytic":
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
            else:
                E = edge_index.shape[1]
                triv_maps = (
                    torch.eye(self.d, device=self.device)
                    .unsqueeze(0)
                    .expand(E, self.d, self.d)
                )
                tmp_builder = lb.GeneralLaplacianBuilder(
                    self.graph_size,
                    edge_index,
                    d=self.d,
                    add_lp=self.add_lp,
                    add_hp=self.add_hp,
                    normalised=self.normalised,
                    deg_normalised=self.deg_normalised,
                )
                (idx, vals), _ = tmp_builder(triv_maps)
                Nd = self.graph_size * self.final_d
                self.lambda_max = estimate_largest_eig(idx, vals, Nd)

        # ---- Linear Maps ----
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight.data)
            self.lin_right_weights.append(r)
        for _ in range(self.layers):
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight.data)
            self.lin_left_weights.append(l)

        # ---- Sheaf Learners (full d×d maps) ----
        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(
                        self.final_d,
                        self.hidden_channels,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                    )
                )
            else:
                self.sheaf_learners.append(
                    self.sheaf_learner(
                        self.hidden_dim,
                        out_shape=(self.d, self.d),
                        sheaf_act=self.sheaf_act,
                        num_edge_types=args.num_edge_types,
                        num_node_types=args.num_node_types,
                    )
                )

        # ---- Laplacian Builder ----
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_lp=self.add_lp,
            add_hp=self.add_hp,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
        )

        # ---- Residual, Embed/Proj, Polynomial Mix ----
        self.epsilons = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros((self.final_d, 1)),
                    requires_grad=getattr(args, "use_epsilons", True),
                )
                for _ in range(self.layers)
            ]
        )
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))  # Softmax Mixture
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

        if self.polynomial_type == "Gegenbauer" and not (self.gc_lambda > 0.0):
            warnings.warn(
                "gegenbauer_lambda must be > 0; clamping to 0.1", stacklevel=2
            )
            self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == "Jacobi" and not (
            self.jac_alpha > -1.0 and self.jac_beta > -1.0
        ):
            warnings.warn(
                "Jacobi requires alpha,beta > -1; clamping to -0.9", stacklevel=2
            )
            self.jac_alpha = max(self.jac_alpha, -0.9)
            self.jac_beta = max(self.jac_beta, -0.9)

    # ---------- L and Lhat ----------
    def _apply_L(self, idx, vals, v):
        return torch_sparse.spmm(idx, vals, v.size(0), v.size(0), v)

    def _apply_Lhat(self, idx, vals, v):
        Lv = self._apply_L(idx, vals, v)
        return (2.0 / self.lambda_max) * Lv - v

    # ---------- Polynomial Evaluation ----------
    def _poly_eval(self, idx, vals, x):
        K = self.K
        w = F.softmax(self.poly_logits, dim=0)

        def Lhat(v):
            return self._apply_Lhat(idx, vals, v)

        def add(acc, k, vec):
            return acc + (w[k] * vec)

        poly = self.polynomial_type

        if poly in ("ChebyshevType1", "Chebyshev"):
            T0 = x
            out = w[0] * T0
            if K >= 1:
                T1 = Lhat(x)
                out = add(out, 1, T1)
                for k in range(1, K):
                    LT1 = Lhat(T1)
                    Tk1 = 2.0 * LT1 - T0
                    out = add(out, k + 1, Tk1)
                    T0, T1 = T1, Tk1
            return out

        if poly == "ChebyshevType2":
            U0 = x
            out = w[0] * U0
            if K >= 1:
                U1 = 2.0 * Lhat(x)
                out = add(out, 1, U1)
                for k in range(1, K):
                    LU1 = Lhat(U1)
                    Uk1 = 2.0 * LU1 - U0
                    out = add(out, k + 1, Uk1)
                    U0, U1 = U1, Uk1
            return out

        if poly == "ChebyshevType3":
            V0 = x
            out = w[0] * V0
            if K >= 1:
                V1 = 2.0 * Lhat(x) - x
                out = add(out, 1, V1)
                for k in range(1, K):
                    LV1 = Lhat(V1)
                    Vk1 = 2.0 * LV1 - V0
                    out = add(out, k + 1, Vk1)
                    V0, V1 = V1, Vk1
            return out

        if poly == "ChebyshevType4":
            W0 = x
            out = w[0] * W0
            if K >= 1:
                W1 = 2.0 * Lhat(x) + x
                out = add(out, 1, W1)
                for k in range(1, K):
                    LW1 = Lhat(W1)
                    Wk1 = 2.0 * LW1 - W0
                    out = add(out, k + 1, Wk1)
                    W0, W1 = W1, Wk1
            return out

        if poly == "Legendre":
            P0 = x
            out = w[0] * P0
            if K >= 1:
                P1 = Lhat(x)
                out = add(out, 1, P1)
                for k in range(1, K):
                    ak = (2.0 * k + 1.0) / (k + 1.0)
                    ck = k / (k + 1.0)
                    LP1 = Lhat(P1)
                    Pk1 = ak * LP1 - ck * P0
                    out = add(out, k + 1, Pk1)
                    P0, P1 = P1, Pk1
            return out

        if poly == "Gegenbauer":
            lam = max(self.gc_lambda, 1e-3)
            C0 = x
            out = w[0] * C0
            if K >= 1:
                C1 = (2.0 * lam) * Lhat(x)
                out = add(out, 1, C1)
                for k in range(1, K):
                    ak = 2.0 * (k + lam) / (k + 1.0)
                    ck = (k + 2.0 * lam - 1.0) / (k + 1.0)
                    LC1 = Lhat(C1)
                    Ck1 = ak * LC1 - ck * C0
                    out = add(out, k + 1, Ck1)
                    C0, C1 = C1, Ck1
            return out

        if poly == "Jacobi":
            a, b = self.jac_alpha, self.jac_beta
            P0 = x
            out = w[0] * P0
            if K >= 1:
                den = a + b + 2.0
                c1 = den / 2.0
                c0 = (a - b) / (den + 0.0)
                P1 = c1 * Lhat(P0) + c0 * P0
                out = add(out, 1, P1)
                for k in range(1, K):
                    den1 = 2.0 * k + a + b
                    den2 = den1 + 2.0
                    Ak = (
                        2.0
                        * (k + 1.0)
                        * (k + a + b + 1.0)
                        / ((den1 + 1.0) * den2 + self._eps)
                    )
                    Bk = (b * b - a * a) / (den1 * den2 + self._eps)
                    Ck = 2.0 * (k + a) * (k + b) / (den1 * (den1 + 1.0) + self._eps)
                    LP1 = Lhat(P1)
                    Pk1 = Ak * LP1 + Bk * P1 - Ck * P0
                    out = add(out, k + 1, Pk1)
                    P0, P1 = P1, Pk1
            return out

        raise ValueError(f"Unknown polynomial_type: {self.polynomial_type}")

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def forward(self, data: Data):
        x = data.x
        # Embedding
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, maps = x, None

        for layer in range(self.layers):
            # (Re)learn Maps
            if layer == 0 or self.nonlinear:
                xm = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                maps = (
                    learner(xm, self.edge_index, maps)
                    if isinstance(learner, RotationInvariantSheafLearner)
                    else learner(xm, self.edge_index, data.edge_type, data.node_type)
                )
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            # Linear Sandwich
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )

            # Polynomial Filter p(L) x
            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)  # for HP skip
            x_poly = self._poly_eval(idx, vals, x)

            # High-pass Skip + Activation + Residual
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            if self.use_act:
                x = F.elu(x)

            x0 = (
                1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
            ) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        return x, maps

    def process_restriction_maps(self, maps: torch.Tensor) -> torch.Tensor:
        return maps

    def __str__(self):
        return f"GeneralSheafPoly-{self.sheaf_type}-{self.polynomial_type}"


#################################################################################################################################
################################################ SheafAN MODELS - Barbero et al.  ###############################################
#################################################################################################################################
class DiscreteSheafAttentionDiffusion(DiscreteSheafDiffusion):
    """Sheaf Attention Network (SheafAN) / Residual SheafAN.

    SheafAN (Eq. 5):     X(t+1) = σ((Λ̂ ⊙ Â_F)(I_n ⊗ W₁)XW₂)
    Res-SheafAN (Eq. 6): X(t+1) = X(t) + σ((Λ̂ ⊙ Â_F - I)(I_n ⊗ W₁)XW₂)

    Args:
        edge_index: Graph edge indices
        args: Configuration object
        sheaf_learner: Sheaf learner type string (reuses existing learners)
    """

    def __init__(
        self,
        edge_index: Tensor,
        args,
        sheaf_learner: str = "type_concat",
    ):
        super(DiscreteSheafAttentionDiffusion, self).__init__(
            edge_index, args, sheaf_learner
        )

        assert self.d > 1, "SheafAN requires d > 1"

        # SheafAN-specific config
        self.heads = getattr(args, "heads", 1)
        self.residual = getattr(args, "residual", False)
        self.orthogonal_maps = getattr(
            args, "orthogonal_maps", True
        )  # In original SheafAN, maps are orthogonal

        # Sheaf adjacency builder (optimized version)
        self.adjacency_builder = san.SheafAdjacencyBuilder(
            num_nodes=self.graph_size,
            edge_index=edge_index,
            d=self.d,
            add_self_loops=True,
            normalised=self.normalised,
        )

        # Precompute attention expansion indices
        self._precompute_attention_indices(edge_index)

        # Sheaf learners
        num_sheaf_learners = self.layers if self.nonlinear else 1
        self.sheaf_learners = nn.ModuleList()
        for _ in range(num_sheaf_learners):
            self.sheaf_learners.append(
                self.sheaf_learner(
                    in_channels=self.hidden_dim,
                    out_shape=(self.d, self.d),
                    sheaf_act=self.sheaf_act,
                    num_node_types=getattr(args, "num_node_types", 1),
                    num_edge_types=getattr(args, "num_edge_types", 1),
                )
            )

        # Attention modules
        self.attention_modules = nn.ModuleList()
        for _ in range(self.layers):
            self.attention_modules.append(
                san.SheafAttention(
                    in_channels=self.hidden_channels,
                    heads=self.heads,
                    dropout=self.dropout,
                    learner_type=self.sheaf_learner,
                    num_node_types=getattr(args, "num_node_types", 1),
                    num_edge_types=getattr(args, "num_edge_types", 1),
                )
            )

        # Weight matrices
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for _ in range(self.layers):
                self.lin_right_weights.append(
                    nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                )
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)

        if self.left_weights:
            for _ in range(self.layers):
                self.lin_left_weights.append(
                    nn.Linear(self.final_d, self.final_d, bias=False)
                )
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        # Residual scaling (only used if residual=True)
        if self.residual:
            self.epsilons = nn.ParameterList()
            for _ in range(self.layers):
                self.epsilons.append(
                    nn.Parameter(
                        torch.zeros((self.final_d, 1)),
                        requires_grad=getattr(args, "use_epsilons", True),
                    )
                )

        # Input/output projections
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def _precompute_attention_indices(self, edge_index: Tensor):
        """Precompute indices for efficient attention expansion."""
        device = edge_index.device
        num_edges = edge_index.size(1)

        # For expanding (E,) attention to (E*d*d,) adjacency values
        edge_idx = torch.arange(num_edges, device=device)
        expanded_edge_idx = (
            edge_idx.view(-1, 1, 1).expand(-1, self.d, self.d).reshape(-1)
        )
        self.register_buffer("_attn_expand_idx", expanded_edge_idx.contiguous())

        # Number of self-loop values for concatenation
        self._num_self_loop_vals = self.graph_size * self.d * self.d

    def left_right_linear(self, x: Tensor, layer: int) -> Tensor:
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = self.lin_left_weights[layer](x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = self.lin_right_weights[layer](x)

        return x

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x = data.x
        edge_type = getattr(data, "edge_type", None)
        node_type = getattr(data, "node_type", None)

        # Input projection
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout if self.second_linear else self.sheaf_dropout,
            training=self.training,
        )

        if self.second_linear:
            x = self.lin12(x)

        x = x.view(self.graph_size * self.final_d, -1)

        if self.residual:
            x0 = x

        maps = None
        adj_data = None

        for layer in range(self.layers):
            # Learn sheaf
            if layer == 0 or self.nonlinear:
                learner_idx = layer if self.nonlinear else 0
                x_maps = F.dropout(
                    x,
                    p=self.sheaf_dropout if layer > 0 else 0.0,
                    training=self.training,
                )
                maps = self.sheaf_learners[learner_idx](
                    x_maps.reshape(self.graph_size, -1),
                    self.edge_index,
                    edge_type,
                    node_type,
                )
                if self.orthogonal_maps:
                    maps = cayley_transform(
                        maps
                    )  # Transform to orthogonal as in original SheafAN paper
                adj_data, transport_maps = self.adjacency_builder(maps)
                self.sheaf_learners[learner_idx].set_L(transport_maps)

            adj_indices, adj_values = adj_data

            # Compute attention
            x_attn = x.reshape(self.graph_size, self.final_d, -1).mean(dim=1)
            alpha = self.attention_modules[layer](
                x_attn,
                self.edge_index,
                self.graph_size,
                edge_type,
                node_type,
            )
            alpha = alpha.mean(dim=1)  # Average heads

            # Direct indexing with precomputed indices
            alpha_expanded = alpha[self._attn_expand_idx]

            if self.adjacency_builder.add_self_loops:
                self_loop_alpha = alpha.new_ones(self._num_self_loop_vals)
                alpha_expanded = torch.cat([alpha_expanded, self_loop_alpha])

            scaled_adj_values = adj_values * alpha_expanded

            # Transforms
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, layer)

            # Diffusion
            x_new = torch_sparse.spmm(
                adj_indices,
                scaled_adj_values,
                self.graph_size * self.d,
                self.graph_size * self.d,
                x,
            )

            if self.use_act:
                x_new = F.elu(x_new)

            # Update
            if self.residual:
                eps = torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
                x0 = (1 + eps) * x0 - x_new
                x = x0
            else:
                x = x_new

        assert torch.all(torch.isfinite(x)), "Numerical instability detected"

        x = x.reshape(self.graph_size, -1)

        return x, maps

    def process_restriction_maps(self, maps: Tensor) -> Tensor:
        return torch.flatten(maps, start_dim=1, end_dim=-1)

    def __str__(self):
        prefix = "ResSheafAN" if self.residual else "SheafAN"
        return f"{prefix}-{self.sheaf_type}"
