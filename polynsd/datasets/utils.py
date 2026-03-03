#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops
import torch


class RemoveSelfLoops(BaseTransform):
    def __init__(self):
        self.attr = "edge_weight"

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or "edge_index" not in store:
                continue

            store.edge_index, store[self.attr] = remove_self_loops(
                store.edge_index,
                edge_attr=store.get(self.attr, None),
            )

        return data


class SubsetTrainSplit(BaseTransform):
    def __init__(
        self, val_ratio: float = 0.1, seed: int = None, use_index: bool = False
    ):
        self.val_ratio = val_ratio
        self.seed = seed
        self.use_index = use_index

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        # Handle both HeteroData and regular Data objects
        node_types = data.node_types if isinstance(data, HeteroData) else [None]

        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed)

        # Flag to ensure we actually found a mask to split
        found_train_mask = False

        for node_type in node_types:
            # Access the store (either data[node_type] or the data object itself)
            store = data[node_type] if node_type is not None else data

            if hasattr(store, "train_mask"):
                found_train_mask = True
                # 1. Convert mask to indices if necessary
                if store.train_mask.dtype == torch.bool:
                    train_indices = torch.where(store.train_mask)[0]
                else:
                    train_indices = store.train_mask

                # 2. Shuffle and Split
                perm = torch.randperm(train_indices.size(0), generator=g)
                train_indices = train_indices[perm]

                val_size = int(train_indices.size(0) * self.val_ratio)
                val_idx = train_indices[:val_size]
                train_idx = train_indices[val_size:]

                # 3. Save as indices or masks
                if self.use_index:
                    store.train_idx = train_idx
                    store.val_idx = val_idx
                    # Optionally remove the old mask to save memory
                    # del store.train_mask
                else:
                    new_train_mask = torch.zeros(store.num_nodes, dtype=torch.bool)
                    new_val_mask = torch.zeros(store.num_nodes, dtype=torch.bool)
                    new_train_mask[train_idx] = True
                    new_val_mask[val_idx] = True
                    store.train_mask = new_train_mask
                    store.val_mask = new_val_mask

        # Raise error if no node type had a training mask
        if not found_train_mask:
            raise ValueError("Data object does not have any 'train_mask' to split.")

        return data
