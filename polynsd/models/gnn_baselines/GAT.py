#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 256,
        heads: int = 8,
        in_channels: int = 64,
        layers: int = 2,
        dropout: float = 0.6,
        input_dropout: float = 0.0,
        initial_dropout: float = 0.0,
        output_dim: int = 256,
        **_kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.initial_dropout = initial_dropout
        self.output_dim = output_dim

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=self.heads,
                dropout=dropout,
                add_self_loops=False,
            )
        )
        # Hidden layers
        for i in range(layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * self.heads,
                    hidden_channels,
                    heads=self.heads,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
        # Last layer with single head
        if layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * self.heads,
                    hidden_channels,
                    heads=1,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )

        # Output projection if needed
        if self.output_dim != hidden_channels:
            self.lin = nn.Linear(hidden_channels, self.output_dim)
        else:
            self.lin = None

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index

        # Input dropout
        x = F.dropout(x, p=self.input_dropout, training=self.training)

        # Layer processing
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            if i == 0:
                x = F.dropout(x, p=self.initial_dropout, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Output projection
        if self.lin is not None:
            x = self.lin(x)

        return x

    def __repr__(self):
        return "GAT"
