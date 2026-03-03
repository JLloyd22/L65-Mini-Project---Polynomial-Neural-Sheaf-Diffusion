import os.path as osp
from typing import Iterable, Optional

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.undirected import to_undirected
try:
    from torch_sparse import coalesce
except Exception:  # pragma: no cover - fallback when torch_sparse is unavailable
    from torch_geometric.utils import coalesce


def _pick_key(keys: Iterable[str], preferred: Optional[Iterable[str]] = None) -> Optional[str]:
    if preferred:
        for key in preferred:
            if key in keys:
                return key
    keys = list(keys)
    if len(keys) == 1:
        return keys[0]
    return keys[0] if keys else None


def _load_mat(path: str, preferred_keys: Optional[Iterable[str]] = None) -> np.ndarray:
    try:
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("__")]
            key = _pick_key(keys, preferred_keys)
            if key is None:
                raise KeyError(f"No data keys found in {path}")
            dataset = f[key]

            if isinstance(dataset, h5py.Group):
                stack = [dataset]
                dataset = None
                while stack:
                    group = stack.pop(0)
                    for subkey in group.keys():
                        item = group[subkey]
                        if isinstance(item, h5py.Dataset):
                            dataset = item
                            stack.clear()
                            break
                        if isinstance(item, h5py.Group):
                            stack.append(item)
                if dataset is None:
                    raise KeyError(f"No dataset found inside group for {path}")
            if dataset.dtype == h5py.ref_dtype:
                refs = dataset[()]
                data = [f[ref][()] for ref in refs.flat]
                data = np.array(data, dtype=object).reshape(refs.shape)
            else:
                data = dataset[()]
            return np.asarray(data)
    except Exception:
        data_dict = sio.loadmat(path)
        keys = [k for k in data_dict.keys() if not k.startswith("__")]
        key = _pick_key(keys, preferred_keys)
        if key is None:
            raise KeyError(f"No data keys found in {path}")
        return np.asarray(data_dict[key])


def _load_h5_cell_series(path: str, preferred_keys: Optional[Iterable[str]] = None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        keys = [k for k in f.keys() if not k.startswith("__")]
        key = _pick_key(keys, preferred_keys)
        if key is None:
            raise KeyError(f"No data keys found in {path}")
        dataset = f[key]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Expected dataset in {path} but found group")
        refs = dataset[()]
        if refs.dtype != h5py.ref_dtype and refs.dtype != object:
            raise TypeError(f"Expected cell-array refs in {path}")
        data = [f[ref][()] for ref in refs.flat]
        return np.array(data, dtype=object)


def _edge_index_from_blist(blist: np.ndarray) -> torch.Tensor:
    arr = np.asarray(blist)
    if arr.ndim == 2 and arr.shape[0] == 2:
        edge_index = arr
    elif arr.ndim == 2 and arr.shape[1] == 2:
        edge_index = arr.T
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        row, col = np.nonzero(arr)
        edge_index = np.vstack([row, col])
    else:
        raise ValueError("blist must be [2, E], [E, 2], or adjacency matrix")

    edge_index = edge_index.astype(np.int64)
    if edge_index.min() == 1:
        edge_index = edge_index - 1
    return torch.tensor(edge_index, dtype=torch.long)


def _normalize_features(x: np.ndarray, n_nodes: int) -> torch.Tensor:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] != n_nodes and x.shape[1] == n_nodes:
        x = x.T
    if x.shape[0] != n_nodes:
        x = x[:n_nodes]
    return torch.tensor(x, dtype=torch.float32)


def _normalize_labels(y: np.ndarray, n_nodes: int) -> torch.Tensor:
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.reshape(-1)
    if y.shape[0] != n_nodes:
        if y.ndim == 2 and y.shape[1] == n_nodes:
            y = y.T.reshape(-1)
        else:
            y = y[:n_nodes]
    y = y.astype(np.int64)
    if y.size > 0 and y.min() == 1:
        y = y - y.min()
    return torch.tensor(y, dtype=torch.long)


def _transpose_if_needed(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        return arr
    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr


def _infer_bus_type(y: torch.Tensor) -> torch.Tensor:
    """
    Infer bus type from the zero pattern in target values (Y_polar format).

    In Y_polar format, zeros indicate KNOWN quantities (inputs to power flow),
    while non-zeros are the UNKNOWN quantities to be predicted:

        PQ  bus (load):      zeros at cols [0,1] → predict v (col 2) and theta (col 3)
        PV  bus (generator): zeros at cols [0,2] → predict q_gen (col 1) and theta (col 3)
        Slack bus (ref):     zeros at cols [2,3] → predict p_gen (col 0) and q_gen (col 1)

    Bus type conventions (0-indexed):
        0 = PQ    — has_v is True  (v is unknown)
        1 = PV    — has_q and not has_v (q_gen unknown, v known)
        2 = Slack — has_p is True  (p_gen is unknown)

    Args:
        y: Target tensor [n_nodes, 4] with cols [p_gen, q_gen, v, theta]
    Returns:
        bus_type: Long tensor [n_nodes] with values in {0, 1, 2}
    """
    has_p = y[:, 0].abs() > 1e-6   # p_gen unknown → Slack
    has_q = y[:, 1].abs() > 1e-6   # q_gen unknown → PV or Slack
    has_v = y[:, 2].abs() > 1e-6   # v unknown     → PQ

    bus_type = torch.zeros(y.shape[0], dtype=torch.long)  # default PQ
    bus_type[has_q & ~has_v] = 1   # PV:    q_gen unknown, v known
    bus_type[has_p]          = 2   # Slack: p_gen unknown
    return bus_type


def _build_loss_mask(bus_type: torch.Tensor, n_dims: int = 4) -> torch.Tensor:
    """
    Build a boolean loss mask [n_nodes, n_dims] indicating which output
    dimensions should be included in the loss for each node.

    Based on Y_polar zero pattern:
        PQ  (type 0): predict cols 2,3 (v, theta)
        PV  (type 1): predict cols 1,3 (q_gen, theta)
        Slack (type 2): predict cols 0,1 (p_gen, q_gen)

    Args:
        bus_type: Long tensor [n_nodes] with values in {0, 1, 2}
        n_dims:   Number of output dimensions (default 4)
    Returns:
        mask: BoolTensor [n_nodes, n_dims]
    """
    n_nodes = bus_type.shape[0]
    mask = torch.zeros(n_nodes, n_dims, dtype=torch.bool)

    pq_idx    = (bus_type == 0).nonzero(as_tuple=True)[0]
    pv_idx    = (bus_type == 1).nonzero(as_tuple=True)[0]
    slack_idx = (bus_type == 2).nonzero(as_tuple=True)[0]

    # PQ buses: predict v (col 2) and theta (col 3)
    mask[pq_idx, 2] = True
    mask[pq_idx, 3] = True

    # PV buses: predict q_gen (col 1) and theta (col 3)
    mask[pv_idx, 1] = True
    mask[pv_idx, 3] = True

    # Slack bus: predict p_gen (col 0) and q_gen (col 1)
    mask[slack_idx, 0] = True
    mask[slack_idx, 1] = True

    return mask


class PowerGridSnapshotDataset:
    def __init__(
        self,
        edge_index: torch.Tensor,
        xs: np.ndarray,
        ys: np.ndarray,
        edge_attr: np.ndarray = None,
    ):
        """
        PowerGridSnapshotDataset stores per-snapshot graph data.

        Normalisation is intentionally NOT applied here — it is handled
        in run.py using training-set statistics only, avoiding data leakage.

        Bus type is inferred on the fly in __getitem__ from the target values
        (p_gen and q_gen columns) and one-hot encoded into the node features.

        Args:
            edge_index: Edge connectivity shared across all snapshots [2, E]
            xs:         List of node feature arrays, one per snapshot
            ys:         List of node target arrays, one per snapshot
            edge_attr:  Optional global edge features shared across snapshots
        """
        self.edge_index = edge_index
        self.xs = list(xs)
        self.ys = list(ys)
        self.edge_attr = (
            torch.tensor(edge_attr, dtype=torch.float32)
            if edge_attr is not None else None
        )
        self.is_snapshot_dataset = True

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a PyG Data object for the snapshot at idx.

        Bus type is inferred from target zeros and one-hot encoded
        as 3 additional columns appended to x:
            [original_features | PQ_flag | PV_flag | Slack_flag]
        """
        x = torch.tensor(self.xs[idx], dtype=torch.float32)
        y = torch.tensor(self.ys[idx], dtype=torch.float32)

        # Infer bus type from structural zeros in targets
        bus_type = _infer_bus_type(y)

        # One-hot encode and append to node features
        bus_onehot = torch.zeros(y.shape[0], 3)
        bus_onehot.scatter_(1, bus_type.unsqueeze(1), 1.0)
        x = torch.cat([x, bus_onehot], dim=1)

        # Build loss mask — True where output dim is unknown (should be predicted)
        loss_mask = _build_loss_mask(bus_type, n_dims=y.shape[1])

        kwargs = {
            'x': x,
            'edge_index': self.edge_index,
            'y': y,
            'bus_type': bus_type,
            'loss_mask': loss_mask,
        }
        if self.edge_attr is not None:
            kwargs['edge_attr'] = self.edge_attr

        return Data(**kwargs)


def load_powergrid_node_dataset(
    name: str,
    root: str,
    target: str = "Y_polar",
) -> PowerGridSnapshotDataset:
    """
    Load the PowerGraph node-level snapshot dataset.

    No normalisation is applied here. All normalisation (z-score of both
    features and targets) is handled in run.py using training-set statistics
    only to avoid data leakage into val/test sets.

    Bus type is inferred per-snapshot in __getitem__ from the target values.
    """
    dataset_root = osp.join(root, name)
    raw_root = osp.join(dataset_root, "raw")

    x_path    = osp.join(raw_root, "X.mat")
    y_path    = osp.join(raw_root, f"{target}.mat")
    edge_path = osp.join(raw_root, "edge_index.mat")

    xs_raw = _load_h5_cell_series(x_path, preferred_keys=["Xpf", "X", "x"])
    ys_raw = _load_h5_cell_series(y_path, preferred_keys=["Y_polarpf", "Y_polar", "y"])

    # Load optional edge attributes (global, shared across snapshots)
    edge_attr_path = osp.join(raw_root, "edge_attr.mat")
    edge_attr_raw = None
    if osp.exists(edge_attr_path):
        edge_attr_raw = _load_mat(edge_attr_path, preferred_keys=["edge_attr"])

    xs = []
    ys = []
    for x_item, y_item in zip(xs_raw, ys_raw):
        x_item = _transpose_if_needed(np.asarray(x_item, dtype=np.float32))
        y_item = _transpose_if_needed(np.asarray(y_item, dtype=np.float32))
        xs.append(x_item)
        ys.append(y_item)

    edge_index = _load_mat(edge_path, preferred_keys=["edge_index", "edgeindex", "edges"])
    edge_index = _edge_index_from_blist(edge_index)
    n_nodes = int(edge_index.max().item()) + 1
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, n_nodes, n_nodes)

    print(f"[load] {name}: {len(xs)} snapshots | {n_nodes} nodes")
    print(f"[load] x shape per snapshot: {xs[0].shape} (before bus-type one-hot)")
    print(f"[load] y shape per snapshot: {ys[0].shape}")

    return PowerGridSnapshotDataset(
        edge_index=edge_index,
        xs=xs,
        ys=ys,
        edge_attr=edge_attr_raw,
    )
    
def load_unified_powergrid_node_dataset(
    root: str,
    target: str = "Y_polar",
    grids: list = None,
) -> "UnifiedPowerGridDataset":
    """
    Load all four PowerGraph node-level datasets and combine them into a
    single unified dataset for cross-grid training.
    """
    if grids is None:
        grids = ["ieee24_node", "ieee39_node", "ieee118_node", "uk_node"]

    all_datasets = []
    for name in grids:
        print(f"[unified] Loading {name}...")
        ds = load_powergrid_node_dataset(name, root=root, target=target)
        all_datasets.append(ds)
        print(f"[unified] {name}: {len(ds)} snapshots, {ds.edge_index.shape[1]} edges")

    return UnifiedPowerGridDataset(all_datasets)


class UnifiedPowerGridDataset:
    """
    Wraps multiple PowerGridSnapshotDatasets into a single iterable dataset.
    Each item is a Data object from one of the constituent grids, preserving
    the grid-specific edge_index, edge_attr, and loss_mask.
    """

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.is_snapshot_dataset = True
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
        self._total = total
        self._data0 = datasets[0][0]
        self.is_snapshot_dataset = True

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        for ds_idx, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                offset = idx - (self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0)
                return self.datasets[ds_idx][offset]
        raise IndexError(f"Index {idx} out of range for UnifiedPowerGridDataset of size {self._total}")

    @property
    def edge_index(self):
        return self._data0.edge_index

    @property
    def edge_attr(self):
        return self._data0.edge_attr if hasattr(self._data0, 'edge_attr') else None

    def get_grid_datasets(self):
        return self.datasets

    def get_grid_for_index(self, idx):
        """Return (grid_dataset, local_idx) for a given global snapshot index."""
        for ds_idx, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                offset = idx - (self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0)
                return self.datasets[ds_idx], offset
        raise IndexError(f"Index {idx} out of range")


def load_powergrid_dataset(
    name: str,
    root: str,
    label_type: str = "bi"
) -> list:
    dataset_root = osp.join(root, name)
    raw_root = osp.join(dataset_root, "raw")

    feature_path = osp.join(raw_root, "Bf.mat")
    edge_path    = osp.join(raw_root, "blist.mat")

    label_file = {
        "bi":  "of_bi.mat",
        "mc":  "of_mc.mat",
        "reg": "of_reg.mat",
    }.get(label_type, "of_bi.mat")
    label_path = osp.join(raw_root, label_file)

    features = _load_mat(feature_path, preferred_keys=["Bf", "bf", "features", "x"])
    blist    = _load_mat(edge_path,    preferred_keys=["blist", "edge_index", "edges"])
    labels   = _load_mat(label_path,   preferred_keys=["of_bi", "of_mc", "of_reg", "labels", "y"])

    edge_index = _edge_index_from_blist(blist)
    n_nodes = int(edge_index.max().item()) + 1

    x = _normalize_features(features, n_nodes)
    y = _normalize_labels(labels, n_nodes)

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, n_nodes, n_nodes)

    data = Data(x=x, edge_index=edge_index, y=y)
    return [data]