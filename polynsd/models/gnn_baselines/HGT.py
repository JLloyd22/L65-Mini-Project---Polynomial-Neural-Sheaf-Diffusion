#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional

import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 256,
        in_channels: Optional[dict[str, int] | DictConfig] = None,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        initial_dropout: float = 0.0,
        output_dim: int = 256,
        **_kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.initial_dropout = initial_dropout
        self.output_dim = output_dim

        # Convert DictConfig/ListConfig to regular Python types if needed
        if isinstance(in_channels, DictConfig):
            in_channels = dict(in_channels) if len(in_channels) > 0 else -1
        elif in_channels is None:
            in_channels = -1
            
        if hasattr(metadata, '__iter__') and not isinstance(metadata, tuple):
            # Convert from Hydra config to regular tuple
            metadata = (list(metadata[0]), [tuple(et) for et in metadata[1]])

        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(
                HGTConv(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    heads=heads,
                    metadata=metadata,
                )
            )

        # Output projection if needed
        if int(self.output_dim) != int(hidden_channels):
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
        for i, layer in enumerate(self.convs):
            x_dict = layer(x_dict, data.edge_index_dict)
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
        return "HGT"
