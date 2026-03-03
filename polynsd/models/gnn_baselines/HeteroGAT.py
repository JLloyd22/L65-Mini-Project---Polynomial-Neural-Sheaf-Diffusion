#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: dict[str, int] | DictConfig,
        hidden_channels: int = 256,
        heads: int = 8,
        layers: int = 2,
        dropout: float = 0.6,
        input_dropout: float = 0.0,
        initial_dropout: float = 0.0,
        output_dim: int = 256,
        target: str = "author",
        **_kwargs,
    ):
        super().__init__()
        self.target = target
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.initial_dropout = initial_dropout
        self.output_dim = output_dim

        # Convert DictConfig/ListConfig to regular Python types if needed
        if isinstance(in_channels, DictConfig):
            in_channels = dict(in_channels)
        if hasattr(metadata, '__iter__') and not isinstance(metadata, tuple):
            # Convert from Hydra config to regular tuple
            metadata = (list(metadata[0]), [tuple(et) for et in metadata[1]])

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            HeteroConv(
                {
                    edge_type: GATConv(
                        in_channels=(
                            in_channels[edge_type[0]],
                            in_channels[edge_type[-1]],
                        ),
                        out_channels=hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                        dropout=dropout,
                    )
                    for edge_type in metadata[1]
                }
            )
        )
        # Middle layers
        for i in range(layers - 2):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        in_channels=hidden_channels * heads,
                        out_channels=hidden_channels,
                        heads=heads,
                        add_self_loops=False,
                        dropout=dropout,
                    )
                    for edge_type in metadata[1]
                }
            )
            self.convs.append(conv)
        # Last layer with single head
        if layers > 1:
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: GATConv(
                            in_channels=hidden_channels * heads,
                            out_channels=hidden_channels,
                            heads=1,
                            add_self_loops=False,
                            dropout=dropout,
                        )
                        for edge_type in metadata[1]
                    }
                )
            )

        # Output projection if needed
        if self.output_dim != hidden_channels:
            self.lin = nn.Linear(hidden_channels, self.output_dim)
        else:
            self.lin = None

    def forward(self, data: HeteroData):
        x_dict = data.x_dict

        # Input dropout
        if self.input_dropout > 0:
            x_dict = {
                key: F.dropout(x, p=self.input_dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # Layer processing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            if i == 0 and self.initial_dropout > 0:
                x_dict = {
                    key: F.dropout(x, p=self.initial_dropout, training=self.training)
                    for key, x in x_dict.items()
                }
            elif i > 0 and self.dropout > 0:
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }

        # Output projection
        if self.lin is not None:
            x_dict = {key: self.lin(x) for key, x in x_dict.items()}

        return x_dict

    def __repr__(self):
        return "HeteroGAT"
