#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_nodes: int,
        num_relations: int,
        layers: int = 2,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        initial_dropout: float = 0.0,
        output_dim: int = 256,
        **_kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.initial_dropout = initial_dropout
        self.output_dim = output_dim

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(RGCNConv(num_nodes, hidden_channels, num_relations))
        # Hidden layers
        for _ in range(layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

        # Output projection if needed
        if self.output_dim != hidden_channels:
            self.lin = nn.Linear(hidden_channels, self.output_dim)
        else:
            self.lin = None

    def forward(self, data: Data):
        edge_type, edge_index = data.edge_type, data.edge_index
        x = None

        # First layer doesn't use input features
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = F.elu(x)
            if i == 0 and self.initial_dropout > 0:
                x = F.dropout(x, p=self.initial_dropout, training=self.training)
            elif i > 0 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Output projection
        if self.lin is not None:
            x = self.lin(x)

        return x

    def __repr__(self):
        return "RGCN"
