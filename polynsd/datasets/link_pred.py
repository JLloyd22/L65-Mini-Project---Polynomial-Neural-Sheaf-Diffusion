#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import Optional, Union

import os
import pandas as pd
import shutil
import torch
import lightning as L
import torch_geometric.transforms as T
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData, Data, download_url, download_google_url, extract_zip
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import MovieLens, LastFM, AmazonBook

from .utils import RemoveSelfLoops

DATA_DIR = "data"


class LinkPredBase(L.LightningDataModule):
    def __init__(
        self,
        target: tuple[str, str, str],
        rev_target: tuple[str, str, str],
        data_dir: str = DATA_DIR,
        is_homogeneous: bool = False,
        num_classes: int = 1,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super(LinkPredBase, self).__init__()
        self.target: tuple[str, str, str] = target
        self.data_dir = data_dir
        self.metadata = None
        self.data = None
        self.pyg_datamodule = None
        self.in_channels = None
        self.num_nodes = None
        self.train_data: Optional[Union[Data, HeteroData]] = None
        self.val_data: Optional[Union[Data, HeteroData]] = None
        self.test_data: Optional[Union[Data, HeteroData]] = None
        self.transform = T.Compose(
            [
                T.Constant(),
                T.ToUndirected(),
                T.NormalizeFeatures(),
                RemoveSelfLoops(),
            ]
        )
        self.rev_target = rev_target
        self.is_homogeneous = is_homogeneous
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.graph_size = 0
        self.task = "binary"
        self.num_node_types: int = 0
        self.num_edge_types: int = 0
        self.num_edges: int = 0
        self.node_type_names: Optional[list[str]] = None
        self.edge_type_names: Optional[list[str]] = None

    def download_data(self) -> HeteroData: ...

    def prepare_data(self) -> None:
        data = self.download_data()

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.num_edges = data.num_edges

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }

        if self.is_homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.num_nodes
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.is_homogeneous else self.target,
            is_undirected=True,
            # split_labels=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=self.rev_target,
        )

        self.train_data, self.val_data, self.test_data = split(data)

        # Recalculate graph_size after split (negative sampling may create new node IDs)
        if self.is_homogeneous:
            max_train = (
                self.train_data.edge_label_index.max().item()
                if self.train_data.edge_label_index.numel() > 0
                else -1
            )
            max_val = (
                self.val_data.edge_label_index.max().item()
                if self.val_data.edge_label_index.numel() > 0
                else -1
            )
            max_test = (
                self.test_data.edge_label_index.max().item()
                if self.test_data.edge_label_index.numel() > 0
                else -1
            )
            self.graph_size = max(
                self.graph_size, max_train + 1, max_val + 1, max_test + 1
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return LightningLinkData(self.train_data, loader="full").full_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.val_data, loader="full").full_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return LightningLinkData(self.test_data, loader="full").full_dataloader()


class LastFMDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super(LastFMDataModule, self).__init__(
            data_dir=f"{data_dir}/lastfm",
            target=("user", "to", "artist"),
            rev_target=("artist", "to", "user"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        data = LastFM(self.data_dir, transform=self.transform)[0]

        del data[self.target]["train_neg_edge_index"]
        del data[self.target]["val_pos_edge_index"]
        del data[self.target]["val_neg_edge_index"]
        del data[self.target]["test_pos_edge_index"]
        del data[self.target]["test_neg_edge_index"]

        return data

    def __repr__(self):
        return "LastFM"


class AmazonBooksDataModule(LinkPredBase):
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        subsample_ratio: float = 1.0,
    ):
        self.subsample_ratio = subsample_ratio
        # Add subsample_ratio to data_dir to force separate cache for different ratios
        if subsample_ratio < 1.0:
            data_dir = f"{data_dir}/amazon_books_subsample_{subsample_ratio}"
        else:
            data_dir = f"{data_dir}/amazon_books"

        super(AmazonBooksDataModule, self).__init__(
            data_dir=data_dir,
            target=("user", "rates", "book"),
            rev_target=("book", "rev_rates", "user"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        # Load from base amazon_books directory (not the subsampled one)
        base_dir = self.data_dir.replace(f"_subsample_{self.subsample_ratio}", "")
        data = AmazonBook(base_dir, transform=self.transform)[0]

        # Subsample edges if requested (for memory efficiency with large graphs)
        if self.subsample_ratio < 1.0:
            # Subsample edges for each edge type
            for edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                num_edges = edge_index.shape[1]
                num_keep = int(num_edges * self.subsample_ratio)

                # Randomly sample edges to keep
                perm = torch.randperm(num_edges)[:num_keep]
                data[edge_type].edge_index = edge_index[:, perm]

            # Remap nodes to contiguous IDs (keep only nodes that have edges), This reduces node count from 144k to whatever nodes are actually used
            for node_type in data.node_types:
                # Find all nodes that appear in edges for this node type
                used_nodes = set()
                for edge_type in data.edge_types:
                    src_type, _, dst_type = edge_type
                    edge_index = data[edge_type].edge_index

                    if src_type == node_type:
                        used_nodes.update(edge_index[0].tolist())
                    if dst_type == node_type:
                        used_nodes.update(edge_index[1].tolist())

                if len(used_nodes) > 0:
                    # Create mapping from old IDs to new contiguous IDs
                    used_nodes = sorted(used_nodes)
                    old_to_new = {old: new for new, old in enumerate(used_nodes)}

                    # Update node features to only include used nodes
                    if data[node_type].x is not None:
                        data[node_type].x = data[node_type].x[used_nodes]

                    # Update num_nodes
                    data[node_type].num_nodes = len(used_nodes)

                    # Remap edge indices
                    for edge_type in data.edge_types:
                        src_type, _, dst_type = edge_type
                        edge_index = data[edge_type].edge_index

                        if src_type == node_type:
                            data[edge_type].edge_index[0] = torch.tensor(
                                [old_to_new[idx.item()] for idx in edge_index[0]]
                            )
                        if dst_type == node_type:
                            data[edge_type].edge_index[1] = torch.tensor(
                                [old_to_new[idx.item()] for idx in edge_index[1]]
                            )

        return data

    def __repr__(self):
        return "AmazonBooks"


class MovieLensDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super(MovieLensDataModule, self).__init__(
            data_dir=f"{data_dir}/movie_lens",
            target=("user", "rates", "movie"),
            rev_target=("movie", "rev_rates", "user"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        data = MovieLens(self.data_dir, transform=self.transform)[0]
        del data[self.target]["edge_label"]
        del data[self.rev_target]["edge_label"]
        # new_edge_labels = torch.ones_like(data[self.target].edge_label)
        # data[self.target].edge_label = new_edge_labels
        # data[self.rev_target].edge_label = new_edge_labels

        # print(data[self.target].edge_label)
        return data

    def __repr__(self):
        return "MovieLens"


class PubMedHNEDataModule(LinkPredBase):
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=f"{data_dir}/pubmed_hne",
            # HNE usually has relations like: 0=P-A, 1=P-C, 2=P-P (citation)
            # Adjust target based on which relation you want to predict.
            # Assuming relation_0 is the target for now.
            target=("node_0", "relation_0", "node_0"),
            rev_target=("node_0", "rev_relation_0", "node_0"),
            is_homogeneous=homogeneous,
        )

    def download_data(self) -> HeteroData:
        # The Drive ID from the link you provided:
        # https://drive.google.com/file/d/1ZEi2sTaZ2bk8cQwyCxtlwuWJsAq9N-Cl/view
        file_id = "1ZEi2sTaZ2bk8cQwyCxtlwuWJsAq9N-Cl"
        zip_name = "pubmed.zip"
        zip_path = os.path.join(self.data_dir, zip_name)

        os.makedirs(self.data_dir, exist_ok=True)

        # 1. Check if files already exist (e.g., node.dat)
        if not os.path.exists(os.path.join(self.data_dir, "node.dat")):
            # 2. Download
            if not os.path.exists(zip_path):
                print(f"Downloading {zip_name} from Google Drive...")
                try:
                    download_google_url(file_id, self.data_dir, zip_name)
                except Exception as e:
                    print(f"PyG download failed ({e}). Trying 'gdown' fallback...")
                    # Fallback to gdown which handles large Drive files/confirmations better
                    try:
                        import gdown
                        url = f'https://drive.google.com/uc?id={file_id}'
                        gdown.download(url, zip_path, quiet=False)
                    except ImportError:
                        raise ImportError(
                            "Download failed. Please install gdown (pip install gdown) "
                            "to handle Google Drive links reliably."
                        )

            # 3. Extract
            if os.path.exists(zip_path):
                print("Extracting...")
                extract_zip(zip_path, self.data_dir)

                # 4. Flatten directory if it extracted into a subdirectory (e.g. 'PubMed/')
                # This ensures node.dat is at data_dir/node.dat, not data_dir/PubMed/node.dat
                possible_subdirs = ["PubMed", "pubmed"]
                for sub in possible_subdirs:
                    subdir_path = os.path.join(self.data_dir, sub)
                    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                        print(f"Moving files from subdirectory {sub}...")
                        for filename in os.listdir(subdir_path):
                            shutil.move(
                                os.path.join(subdir_path, filename), 
                                os.path.join(self.data_dir, filename)
                            )
                        os.rmdir(subdir_path)
        
        # Load and return the heterogeneous data
        return self._load_heterogeneous_data()

    def _load_heterogeneous_data(self) -> HeteroData:
        """Load and return heterogeneous data from pubmed files (optimized)."""
        # Parse node.dat
        node_file = os.path.join(self.data_dir, "node.dat")
        if not os.path.exists(node_file):
            raise FileNotFoundError(f"Could not find {node_file}. Download or extraction failed.")

        node_df = pd.read_csv(
            node_file, 
            sep="\t", header=None, names=["id", "name", "type", "misc"],
            dtype={"id": int, "type": int}
        )
        
        # Create mappings efficiently - O(n) operation
        node_types = sorted(node_df["type"].unique())
        type_counters = {t: 0 for t in node_types}
        global_to_local = {}
        
        # Single pass to build mapping
        for node_id, node_type in zip(node_df["id"].values, node_df["type"].values):
            global_to_local[node_id] = (node_type, type_counters[node_type])
            type_counters[node_type] += 1
        
        counts = type_counters
        
        # Load links efficiently
        link_file = os.path.join(self.data_dir, "link.dat")
        if not os.path.exists(link_file):
            raise FileNotFoundError(f"Could not find {link_file}. Download or extraction failed.")

        link_df = pd.read_csv(
            link_file, sep="\t", header=None, names=["h", "t", "r", "w"],
            dtype={"h": int, "t": int, "r": int}
        )
        
        data = HeteroData()
        
        # Initialize nodes with random features
        for t in node_types:
            t_name = f"node_{t}" 
            data[t_name].num_nodes = counts[t]
            data[t_name].x = torch.randn(counts[t], 64)
        
        # Filter valid edges and process by relation
        valid_mask = link_df["h"].isin(global_to_local.keys()) & link_df["t"].isin(global_to_local.keys())
        valid_links = link_df[valid_mask]
        
        for r_id, group in valid_links.groupby("r"):
            # Map global IDs to local indices using vectorized lookup
            h_vals = group["h"].values
            t_vals = group["t"].values
            
            src_data = [global_to_local[h] for h in h_vals]
            dst_data = [global_to_local[t] for t in t_vals]
            
            src_indices = [s[1] for s in src_data]
            dst_indices = [d[1] for d in dst_data]
            src_type = src_data[0][0]
            dst_type = dst_data[0][0]
            
            # Create edge tensors
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            
            rel_name = f"relation_{r_id}"
            src_name = f"node_{src_type}"
            dst_name = f"node_{dst_type}"
            
            data[src_name, rel_name, dst_name].edge_index = edge_index
            
            # Add reverse edges
            rev_name = f"rev_relation_{r_id}"
            data[dst_name, rev_name, src_name].edge_index = edge_index.flip(0)
        
        return data

    def prepare_data(self) -> None:
        self.download_data()
        
        # Use parent's prepare_data to handle metadata and splitting
        data = self.download_data()

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.num_edges = data.num_edges

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }

        if self.is_homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.num_nodes
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.is_homogeneous else self.target,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=self.rev_target,
        )

        self.train_data, self.val_data, self.test_data = split(data)

        # Recalculate graph_size after split (negative sampling may create new node IDs)
        if self.is_homogeneous:
            max_train = (
                self.train_data.edge_label_index.max().item()
                if self.train_data.edge_label_index.numel() > 0
                else -1
            )
            max_val = (
                self.val_data.edge_label_index.max().item()
                if self.val_data.edge_label_index.numel() > 0
                else -1
            )
            max_test = (
                self.test_data.edge_label_index.max().item()
                if self.test_data.edge_label_index.numel() > 0
                else -1
            )
            self.graph_size = max(
                self.graph_size, max_train + 1, max_val + 1, max_test + 1
            )

    def __repr__(self):
        return "PubMedHNE"

class AmazonGATNEDataModule(LinkPredBase):
    """
    Amazon dataset from 'Representation learning for attributed multiplex heterogeneous network' 
    (Cen et al. 2019, KDD).
    
    Nodes: Items (all same type)
    Edges: Multiple types (also_viewed, also_bought, etc.)
    """
    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(
            data_dir=f"{data_dir}/amazon_gatne",
            target=("item", "relation_0", "item"), # Primary target relation
            rev_target=("item", "rev_relation_0", "item"),
            is_homogeneous=homogeneous,
        )
        self.raw_url = "https://github.com/THUDM/GATNE/raw/master/data/amazon/"

    def download_data(self) -> HeteroData:
        files = ["train.txt", "valid.txt", "test.txt"]
        os.makedirs(self.data_dir, exist_ok=True)
        for f in files:
            path = os.path.join(self.data_dir, f)
            if not os.path.exists(path):
                print(f"Downloading {f}...")
                download_url(self.raw_url + f, self.data_dir)
        
        # Load and return the heterogeneous data from training set
        return self._load_amazon_gatne_data()

    def _load_amazon_gatne_data(self) -> HeteroData:
        """Load and return heterogeneous data from Amazon GATNE files (optimized)."""
        # Read all files efficiently - just get unique nodes
        unique_nodes = set()
        for f in ["train.txt", "valid.txt", "test.txt"]:
            file_path = os.path.join(self.data_dir, f)
            if os.path.exists(file_path):
                # Read only the node columns, skip type
                df = pd.read_csv(
                    file_path, sep=" ", header=None, usecols=[0, 1],
                    names=["u", "v"], dtype=str
                )
                unique_nodes.update(df["u"].values)
                unique_nodes.update(df["v"].values)
        
        # Build node map - deterministic ordering
        node_map = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
        num_nodes = len(node_map)

        # Load training data once
        train_df = pd.read_csv(
            os.path.join(self.data_dir, "train.txt"), sep=" ", header=None, 
            names=["u", "v", "type"], dtype={"u": str, "v": str, "type": int}
        )

        data = HeteroData()
        data["item"].num_nodes = num_nodes
        data["item"].x = torch.randn(num_nodes, 128)

        # Vectorized mapping - use map() instead of list comprehension
        for t_id, group in train_df.groupby("type"):
            # Vectorized lookup using pandas map (much faster than list comprehension)
            src = torch.tensor(group["u"].map(node_map).values, dtype=torch.long)
            dst = torch.tensor(group["v"].map(node_map).values, dtype=torch.long)
            
            rel_name = f"relation_{t_id}"
            rev_rel_name = f"rev_relation_{t_id}"
            
            data["item", rel_name, "item"].edge_index = torch.stack([src, dst])
            data["item", rev_rel_name, "item"].edge_index = torch.stack([dst, src])
        
        return data

    def prepare_data(self) -> None:
        data = self.download_data()

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.num_edges = data.num_edges

        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }

        if self.is_homogeneous:
            data = data.to_homogeneous()
            self.graph_size = data.num_nodes
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.is_homogeneous else self.target,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=self.rev_target,
        )

        self.train_data, self.val_data, self.test_data = split(data)

        # Recalculate graph_size after split (negative sampling may create new node IDs)
        if self.is_homogeneous:
            max_train = (
                self.train_data.edge_label_index.max().item()
                if self.train_data.edge_label_index.numel() > 0
                else -1
            )
            max_val = (
                self.val_data.edge_label_index.max().item()
                if self.val_data.edge_label_index.numel() > 0
                else -1
            )
            max_test = (
                self.test_data.edge_label_index.max().item()
                if self.test_data.edge_label_index.numel() > 0
                else -1
            )
            self.graph_size = max(
                self.graph_size, max_train + 1, max_val + 1, max_test + 1
            )

    def __repr__(self):
        return "AmazonGATNE"

# class AmazonGATNEDataModule(LinkPredBase):
#     def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
#         super().__init__(
#             data_dir=f"{data_dir}/amazon_gatne",
#             target=("item", "relation_0", "item"),
#             rev_target=("item", "rev_relation_0", "item"),
#             is_homogeneous=homogeneous,
#         )
#         self.in_channels = {"item": 128}
#         self.raw_url = "https://github.com/THUDM/GATNE/raw/master/data/amazon/"

#     def download_data(self) -> None:
#         files = ["train.txt", "valid.txt", "test.txt"]
#         os.makedirs(self.data_dir, exist_ok=True)
#         for f in files:
#             path = os.path.join(self.data_dir, f)
#             if not os.path.exists(path):
#                 print(f"Downloading {f}...")
#                 download_url(self.raw_url + f, self.data_dir)

#     def prepare_data(self) -> None:
#         # 1. Check for cached processed data
#         processed_path = os.path.join(self.data_dir, "processed_data.pt")
#         if os.path.exists(processed_path):
#             print("Loading cached Amazon data (fast)...")
#             data_dict = torch.load(processed_path, weights_only=False)
#             self.train_data = data_dict["train"]
#             self.val_data = data_dict["val"]
#             self.test_data = data_dict["test"]
#             self.num_nodes = data_dict["num_nodes"]
#             return

#         print("Processing Amazon data from scratch (this may take a minute)...")
#         self.download_data()

#         # 2. Build Global Node Map (Vectorized)
#         # Read all files into one dataframe to find unique nodes
#         print("Reading raw files...")
#         dfs = []
#         for f in ["train.txt", "valid.txt", "test.txt"]:
#             df = pd.read_csv(os.path.join(self.data_dir, f), sep=" ", header=None, names=["u", "v", "type"])
#             dfs.append(df)
        
#         full_df = pd.concat(dfs, ignore_index=True)
        
#         # Get unique nodes and sort them
#         unique_nodes = pd.unique(full_df[["u", "v"]].values.ravel('K'))
#         unique_nodes.sort()
        
#         node_map = pd.Series(index=unique_nodes, data=range(len(unique_nodes)))
#         self.num_nodes = len(unique_nodes)
#         print(f"Found {self.num_nodes} unique nodes.")

#         # 3. Helper to process splits using Vectorization
#         def process_split(filename):
#             print(f"Processing {filename}...")
#             df = pd.read_csv(os.path.join(self.data_dir, filename), sep=" ", header=None, names=["u", "v", "type"])
            
#             # OPTIMIZATION: Map strings to integers using the pandas Series (very fast)
#             df["u_idx"] = df["u"].map(node_map)
#             df["v_idx"] = df["v"].map(node_map)
            
#             data = HeteroData()
#             data["item"].num_nodes = self.num_nodes
#             data["item"].x = torch.randn(self.num_nodes, 128) # Random features

#             # Group by edge type
#             for t_id, group in df.groupby("type"):
#                 # Convert directly from numpy values (faster than list comprehension)
#                 src = torch.from_numpy(group["u_idx"].values.astype('int64'))
#                 dst = torch.from_numpy(group["v_idx"].values.astype('int64'))
                
#                 rel_name = f"relation_{t_id}"
#                 rev_rel_name = f"rev_relation_{t_id}"
                
#                 data["item", rel_name, "item"].edge_index = torch.stack([src, dst])
                
#                 if "train" in filename:
#                     data["item", rev_rel_name, "item"].edge_index = torch.stack([dst, src])
            
#             return data

#         # 4. Process and Cache
#         self.train_data = process_split("train.txt")
#         self.val_data = process_split("valid.txt")
#         self.test_data = process_split("test.txt")

#         if self.is_homogeneous:
#             self.train_data = self.train_data.to_homogeneous()
#             self.val_data = self.val_data.to_homogeneous()
#             self.test_data = self.test_data.to_homogeneous()

#         # Save to disk
#         print("Saving processed data to disk...")
#         torch.save({
#             "train": self.train_data,
#             "val": self.val_data,
#             "test": self.test_data,
#             "num_nodes": self.num_nodes
#         }, processed_path)
#         print("Done!")

#     def __repr__(self):
#         return "AmazonGATNE"