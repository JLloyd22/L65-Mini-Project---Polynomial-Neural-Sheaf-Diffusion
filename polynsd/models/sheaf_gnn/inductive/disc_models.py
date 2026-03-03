#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from torch_geometric.data import Data

from polynsd.models.sheaf_gnn.transductive.disc_models import (
    DiscreteDiagSheafDiffusion,
    DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
    DiscreteDiagSheafDiffusionPolynomial,
    DiscreteBundleSheafDiffusionPolynomial,
    DiscreteGeneralSheafDiffusionPolynomial,
    DiscreteSheafAttentionDiffusion,
)
from polynsd.models.sheaf_gnn import sheaf_attention as san

#TODO: refactor the name where Polynomial is put before Discrete to match other files
class InductiveDiscreteDiagSheafDiffusion(DiscreteDiagSheafDiffusion):
    def __init__(self, args, sheaf_learner):
        super(InductiveDiscreteDiagSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data: Data):
        # Update graph size for variable-sized graphs
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteDiagSheafDiffusion, self).forward(data)


class InductiveDiscreteBundleSheafDiffusion(DiscreteBundleSheafDiffusion):
    def __init__(self, args, sheaf_learner):
        super(InductiveDiscreteBundleSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteBundleSheafDiffusion, self).forward(data)


class InductiveDiscreteGeneralSheafDiffusion(DiscreteGeneralSheafDiffusion):
    def __init__(self, args, sheaf_learner):
        super(InductiveDiscreteGeneralSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteGeneralSheafDiffusion, self).forward(data)


### -------------------------------------- Polynomial Sheaf Diffusion Models -------------------------------------- ###
class InductivePolynomialDiscreteDiagSheafDiffusion(
    DiscreteDiagSheafDiffusionPolynomial
):
    def __init__(self, args, sheaf_learner):
        super(InductivePolynomialDiscreteDiagSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductivePolynomialDiscreteDiagSheafDiffusion, self).forward(data)


class InductivePolynomialDiscreteBundleSheafDiffusion(
    DiscreteBundleSheafDiffusionPolynomial
):
    def __init__(self, args, sheaf_learner):
        super(InductivePolynomialDiscreteBundleSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductivePolynomialDiscreteBundleSheafDiffusion, self).forward(
            data
        )


class InductivePolynomialDiscreteGeneralSheafDiffusion(
    DiscreteGeneralSheafDiffusionPolynomial
):
    def __init__(self, args, sheaf_learner):
        super(InductivePolynomialDiscreteGeneralSheafDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def forward(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductivePolynomialDiscreteGeneralSheafDiffusion, self).forward(
            data
        )


## -------------------------------------- Sheaf Attention Network - Barbero et al. -------------------------------------- ##


class InductiveDiscreteSheafAttentionDiffusion(DiscreteSheafAttentionDiffusion):
    def __init__(self, args, sheaf_learner):
        super(InductiveDiscreteSheafAttentionDiffusion, self).__init__(
            edge_index=None, args=args, sheaf_learner=sheaf_learner
        )

    def _precompute_attention_indices(self, edge_index):
        """Override to handle None edge_index for inductive learning."""
        if edge_index is None:
            # Skip precomputation when edge_index is None (will be done in update_edge_index)
            return
        super()._precompute_attention_indices(edge_index)

    def update_edge_index(self, edge_index):
        """Update edge index for inductive learning (uses adjacency_builder)."""
        assert edge_index.max() <= self.graph_size
        self.edge_index = edge_index
        # Precompute attention indices for this edge configuration
        self._precompute_attention_indices(edge_index)
        # For attention diffusion, update the adjacency builder

        self.adjacency_builder.update_graph(num_nodes=self.graph_size, edge_index=edge_index)
        #self.adjacency_builder = san.SheafAdjacencyBuilder(
        #    num_nodes=self.graph_size,
        #    edge_index=edge_index,
        #    d=self.d,
        #    add_self_loops=True,
        #    normalised=self.normalised,
        #)

    def forward(self, data: Data):
        num_nodes = data.x.size(0)
        if num_nodes != self.graph_size:
            self.graph_size = num_nodes
        self.update_edge_index(data.edge_index)
        return super(InductiveDiscreteSheafAttentionDiffusion, self).forward(data)
