import sys
import os
import random
import time
import numpy as np
import torch.optim.lr_scheduler
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

# Setup compatibility layers for optional dependencies before importing torch-dependent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try importing torch_sparse, fall back to compat layer
try:
    import torch_sparse
except ImportError:
    from torch_sparse_compat import spmm
    
    class TorchSparseCompat:
        spmm = staticmethod(spmm)
    
    sys.modules['torch_sparse'] = TorchSparseCompat()

# Try importing torch_scatter, fall back to compat layer
try:
    import torch_scatter
except ImportError:
    from torch_scatter_compat import scatter_add, scatter
    
    class TorchScatterCompat:
        scatter_add = staticmethod(scatter_add)
        scatter = staticmethod(scatter)
    
    sys.modules['torch_scatter'] = TorchScatterCompat()

# Try importing torch_householder, fall back to compat layer
try:
    import torch_householder
except ImportError:
    from torch_householder_compat import torch_householder_orgqr
    
    class TorchHouseholderCompat:
        torch_householder_orgqr = staticmethod(torch_householder_orgqr)
    
    sys.modules['torch_householder'] = TorchHouseholderCompat()

import torch
import torch.nn.functional as F
import git

import wandb
from tqdm import tqdm
from torch_geometric.data import Data

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import (
    DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion,
    DiagSheafDiffusion_Polynomial, BundleSheafDiffusion_Polynomial, GeneralSheafDiffusion_Polynomial
)
from models.disc_models import (
    DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion,
    DiscreteDiagSheafDiffusionPolynomial, DiscreteBundleSheafDiffusionPolynomial,
    DiscreteGeneralSheafDiffusionPolynomial, PolySpectralGNN
)
from utils.heterophilic import get_dataset, get_synthetic_dataset, get_fixed_splits

# reproducibility utilities (we will use them ONLY in resource_analysis mode)
from utils.reproducibility import set_reproducible, fold_seed, truthy

# resource analysis utilities (only used when resource_analysis=True)
from utils.resource_analysis import (
    ResourceMonitor,
    device_cuda_index,
    profiler_available,
    train_step_with_optional_flops,
    maybe_profile_macs_torchprofile,
)


# ----------------------------- save directory helpers -----------------------------
def get_save_dir(base_name: str = "outputs") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_predictions(preds: torch.Tensor, fold: int, tag: str = "U_matrix") -> str:
    save_dir = get_save_dir("outputs")
    save_path = os.path.join(save_dir, f"{tag}_fold{fold}.pt")
    torch.save(preds, save_path)
    print(f"[save] Saved {tag} for fold {fold} to: {save_path}")
    return save_path


def save_node_embeddings(model: torch.nn.Module, data: Data, fold: int, tag: str = "node_embeddings") -> Optional[str]:
    save_dir = get_save_dir("outputs")
    save_path = os.path.join(save_dir, f"{tag}_fold{fold}.pt")
    try:
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x)
        torch.save(embeddings.detach().cpu(), save_path)
        print(f"[save] Saved node embeddings for fold {fold} to: {save_path}")
        return save_path
    except Exception as e:
        print(f"[warn] Could not save node embeddings for fold {fold}: {e}")
        return None


def save_umap_checkpoint(
    model: torch.nn.Module,
    data: Data,
    labels: torch.Tensor,
    fold: int,
    epoch: int,
    umap_reducer=None,
) -> Optional[object]:
    import matplotlib.pyplot as plt
    import io

    try:
        from umap import UMAP
    except ImportError:
        print("[warn] umap-learn not installed — skipping UMAP checkpoint.")
        return umap_reducer

    try:
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()

        emb_tensor = torch.tensor(embeddings)
        save_predictions(emb_tensor, fold, tag=f"node_embeddings_epoch{epoch}")

        if umap_reducer is None:
            umap_reducer = UMAP(n_components=2, random_state=42)
            coords = umap_reducer.fit_transform(embeddings)
        else:
            coords = umap_reducer.transform(embeddings)

        n_targets = labels_np.shape[1] if labels_np.ndim > 1 else 1
        has_v_mag = n_targets >= 1
        has_theta = n_targets >= 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if has_v_mag:
            sc0 = axes[0].scatter(
                coords[:, 0], coords[:, 1],
                c=labels_np[:, 0], cmap='viridis', s=8, alpha=0.8
            )
            plt.colorbar(sc0, ax=axes[0], label='V Magnitude')
            axes[0].set_title(f'UMAP — V Magnitude | Fold {fold} Epoch {epoch}')
            axes[0].set_xlabel('UMAP 1')
            axes[0].set_ylabel('UMAP 2')

        if has_theta:
            sc1 = axes[1].scatter(
                coords[:, 0], coords[:, 1],
                c=labels_np[:, -1], cmap='plasma', s=8, alpha=0.8
            )
            plt.colorbar(sc1, ax=axes[1], label='Theta')
            axes[1].set_title(f'UMAP — Theta | Fold {fold} Epoch {epoch}')
            axes[1].set_xlabel('UMAP 1')
            axes[1].set_ylabel('UMAP 2')

        plt.tight_layout()

        umap_dir = get_save_dir("outputs/umaps")
        fig_path = os.path.join(umap_dir, f"umap_fold{fold}_epoch{epoch}.png")
        fig.savefig(fig_path, dpi=120)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({
            f"fold{fold}_umap": wandb.Image(buf, caption=f"Fold {fold} Epoch {epoch}")
        }, step=epoch)
        plt.close(fig)

        print(f"[save] UMAP checkpoint saved for fold {fold} epoch {epoch} to: {fig_path}")
        return umap_reducer

    except Exception as e:
        print(f"[warn] UMAP checkpoint failed for fold {fold} epoch {epoch}: {e}")
        return umap_reducer


# ----------------------------- restriction map helpers -----------------------------

_RUN_DIR: Optional[Path] = None


def _get_run_dir() -> Path:
    global _RUN_DIR
    if _RUN_DIR is not None:
        return _RUN_DIR

    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    slurm_id = os.environ.get("SLURM_JOB_ID", None)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{slurm_id}_{timestamp}" if slurm_id else f"local_{timestamp}"

    _RUN_DIR = outputs_dir / folder_name
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Run output directory: {_RUN_DIR}")
    return _RUN_DIR


def _collect_restriction_maps(model) -> dict:
    maps = {}
    sheaf_learners = getattr(model, "sheaf_learners", None)
    if sheaf_learners is None:
        return maps
    for idx, learner in enumerate(sheaf_learners):
        layer_maps = getattr(learner, "L", None)
        if layer_maps is not None:
            maps[f"layer_{idx}"] = layer_maps.detach().cpu()
    return maps


def save_restriction_maps(model, fold: int, epoch: int) -> None:
    try:
        run_dir = _get_run_dir()
        maps = _collect_restriction_maps(model)
        if not maps:
            print(f"[warn] No restriction maps found on model for fold {fold} epoch {epoch}.")
            return
        for layer_name, layer_tensor in maps.items():
            save_dir = run_dir / "restriction_maps" / f"fold{fold}" / layer_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"epoch{epoch}.pt"
            torch.save(layer_tensor, save_path)
            print(f"[save] Restriction map {layer_name} fold {fold} epoch {epoch} → {save_path}")
    except Exception as e:
        print(f"[warn] Could not save restriction maps for fold {fold} epoch {epoch}: {e}")


def save_model_artifacts(model, args, fold: int) -> None:
    try:
        run_dir = _get_run_dir()
        dataset_name = str(aget(args, "dataset", "unknown")).lower().replace(" ", "_")
        model_name = str(aget(args, "model", "sheaf"))

        state_dict = OrderedDict(
            (k, v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in model.state_dict().items()
        )
        checkpoint = {
            "state_dict": state_dict,
            "metadata": {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold,
                "saved_at": datetime.now().isoformat(),
            },
        }
        torch.save(checkpoint, run_dir / f"model_fold{fold}.pt")

        maps = _collect_restriction_maps(model)
        if maps:
            maps_dir = run_dir / "restriction_maps" / f"fold{fold}"
            maps_dir.mkdir(parents=True, exist_ok=True)
            for layer_name, layer_tensor in maps.items():
                save_path = maps_dir / f"{layer_name}.pt"
                torch.save(layer_tensor, save_path)
                print(f"[save] Restriction map {layer_name} fold {fold} → {save_path}")
        else:
            print(f"[warn] No restriction maps found on model for fold {fold}.")

        print(f"[save] Model artifacts for fold {fold} saved to {run_dir}")
    except Exception as exc:
        print(f"[warn] Failed to save model artifacts for fold {fold}: {exc}")


# ----------------------------- helpers -----------------------------
def aget(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    try:
        return args[key]
    except Exception:
        return getattr(args, key, default)


def normalize_device(dev):
    if isinstance(dev, torch.device):
        return dev
    return torch.device(str(dev))


# ----------------------------- train / eval -----------------------------
def train(model, optimizer, data, task: str = "classification"):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    if task == "regression":
        if hasattr(data, "train_mask"):
            out = out[data.train_mask]
            target = data.y[data.train_mask]
        else:
            target = data.y
        loss = F.mse_loss(out, target)
    else:
        out = out[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    del out


def test(model, data, task: str = "classification"):
    model.eval()
    with torch.no_grad():
        logits = model(data.x)
        accs, losses, preds = [], [], []
        for _, mask in data("train_mask", "val_mask", "test_mask"):
            if task == "regression":
                target = data.y[mask]
                loss = F.mse_loss(logits[mask], target)
                accs.append(None)
                losses.append(loss.detach().cpu())
                preds.append(logits[mask].detach().cpu())
            else:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(logits[mask], data.y[mask])
                preds.append(pred.detach().cpu())
                accs.append(acc)
                losses.append(loss.detach().cpu())
        return accs, preds, losses


def is_snapshot_dataset(dataset) -> bool:
    return bool(getattr(dataset, "is_snapshot_dataset", False))


def split_snapshot_indices(n: int, seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _per_dim_mse(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = out - target
    if diff.dim() == 1:
        diff = diff.view(-1, 1)
    return (diff ** 2).mean(dim=0)


def _unmasked_losses(out: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor = None):
    if target.dim() == 1:
        target = target.view(-1, 1)
        out = out.view(-1, 1)

    if mask is None:
        diff = out - target
        loss = (diff ** 2).mean()
        mae = diff.abs().mean()
        rmse = torch.sqrt(loss)
        per_dim_mse = (diff ** 2).mean(dim=0)
        per_dim_mae = diff.abs().mean(dim=0)
    else:
        if mask.dim() == 1:
            mask = mask.view(-1, 1)
        mask = mask.to(out.device)
        n_valid = mask.sum().clamp(min=1)
        sq_err = ((out - target) ** 2) * mask
        abs_err = (out - target).abs() * mask
        loss = sq_err.sum() / n_valid
        mae = abs_err.sum() / n_valid
        rmse = torch.sqrt(loss)
        n_valid_per_dim = mask.sum(dim=0).clamp(min=1)
        per_dim_mse = sq_err.sum(dim=0) / n_valid_per_dim
        per_dim_mae = abs_err.sum(dim=0) / n_valid_per_dim

    return loss, mae, rmse, per_dim_mse, per_dim_mae


def _compute_normalisation(dataset, train_idx):
    train_data = [dataset[int(i)] for i in train_idx]

    xs = torch.stack([d.x for d in train_data], dim=0).float()
    x_raw = xs[:, :, :-3]
    x_mean = x_raw.mean(dim=0)
    x_std  = x_raw.std(dim=0)

    col_std = x_std.mean(dim=0)
    const_x_mask = col_std < 1e-3

    print(f"[normalise] col_std (avg per feature): {col_std.tolist()}")
    print(f"[normalise] const_x_mask: {const_x_mask.tolist()}")

    if const_x_mask.any():
        const_cols = const_x_mask.nonzero(as_tuple=True)[0].tolist()
        print(f"[normalise] WARNING: x columns {const_cols} have near-zero variance "
              f"(avg std < 1e-3) — these will not be z-scored to avoid instability")

    x_std_safe = x_std.clone()
    x_std_safe[:, const_x_mask] = 1.0
    x_std_safe = x_std_safe.clamp(min=1e-6)
    x_std_safe[:, const_x_mask] = 1.0

    ys = torch.stack([d.y for d in train_data], dim=0).float()
    y_mean = ys.mean(dim=0)
    y_std  = ys.std(dim=0).clamp(min=1e-2)

    print(f"[normalise] x_mean per dim (avg over nodes): {x_mean.mean(dim=0).tolist()}")
    print(f"[normalise] x_std  per dim (avg over nodes): {x_std.mean(dim=0).tolist()}")
    print(f"[normalise] y_mean per dim (avg over nodes): {y_mean.mean(dim=0).tolist()}")
    print(f"[normalise] y_std  per dim (avg over nodes): {y_std.mean(dim=0).tolist()}")

    return x_mean, x_std_safe, y_mean, y_std, const_x_mask


def _normalise_data(data, x_mean, x_std, y_mean, y_std, const_x_mask=None):
    data = data.clone()
    x = data.x.float()

    n_raw = x_mean.shape[1]
    x_raw    = x[:, :n_raw]
    x_onehot = x[:, n_raw:]

    x_raw_norm = (x_raw - x_mean.to(x.device)) / x_std.to(x.device)
    data.x = torch.cat([x_raw_norm, x_onehot], dim=1)
    data.y = (data.y.float() - y_mean.to(data.y.device)) / y_std.to(data.y.device)
    return data


def _denormalise_preds(preds, y_mean, y_std):
    return preds.float() * y_std.cpu() + y_mean.cpu()


def train_snapshot_batch(model, optimizer, x_batch: torch.Tensor, y_batch: torch.Tensor,
                         edge_attr: torch.Tensor = None, mask_batch: torch.Tensor = None):
    model.train()
    optimizer.zero_grad()

    batch_size, n_nodes, x_dims = x_batch.shape
    y_dims = y_batch.shape[2]

    all_out = []
    for i in range(batch_size):
        out_i = model(x_batch[i], edge_attr=edge_attr)
        all_out.append(out_i)

    out = torch.stack(all_out, dim=0)

    mask_flat = mask_batch.reshape(batch_size * n_nodes, y_dims) if mask_batch is not None else None
    loss, mae, rmse, per_dim_mse, per_dim_mae = _unmasked_losses(
        out.reshape(batch_size * n_nodes, y_dims),
        y_batch.reshape(batch_size * n_nodes, y_dims),
        mask=mask_flat,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return (
        float(loss.detach().cpu()),
        float(mae.detach().cpu()),
        float(rmse.detach().cpu()),
        per_dim_mse.detach().cpu().numpy(),
        per_dim_mae.detach().cpu().numpy(),
    )


def eval_snapshot_batch(model, x_batch: torch.Tensor, y_batch: torch.Tensor,
                        edge_attr: torch.Tensor = None, mask_batch: torch.Tensor = None):
    model.eval()
    with torch.no_grad():
        batch_size, n_nodes, x_dims = x_batch.shape
        y_dims = y_batch.shape[2]

        all_out = []
        for i in range(batch_size):
            out_i = model(x_batch[i], edge_attr=edge_attr)
            all_out.append(out_i)

        out = torch.stack(all_out, dim=0)

        mask_flat = mask_batch.reshape(batch_size * n_nodes, y_dims) if mask_batch is not None else None
        loss, mae, rmse, per_dim_mse, per_dim_mae = _unmasked_losses(
            out.reshape(batch_size * n_nodes, y_dims),
            y_batch.reshape(batch_size * n_nodes, y_dims),
            mask=mask_flat,
        )
    return (
        float(loss.detach().cpu()),
        float(mae.detach().cpu()),
        float(rmse.detach().cpu()),
        per_dim_mse.detach().cpu().numpy(),
        per_dim_mae.detach().cpu().numpy(),
    )


def train_snapshot(model, optimizer, data, edge_attr=None):
    x_batch = data.x.unsqueeze(0)
    y_batch = data.y.unsqueeze(0)
    return train_snapshot_batch(model, optimizer, x_batch, y_batch, edge_attr=edge_attr)


def eval_snapshot(model, data, edge_attr=None) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    x_batch = data.x.unsqueeze(0)
    y_batch = data.y.unsqueeze(0)
    return eval_snapshot_batch(model, x_batch, y_batch, edge_attr=edge_attr)


# =====================================================================
#  CLASSIC FOLD RUN
# =====================================================================
def run_exp_classic(args, dataset, model_cls, fold: int) -> Tuple[float, float, bool]:
    if is_snapshot_dataset(dataset):
        return run_exp_snapshot_classic(args, dataset, model_cls, fold)

    data = dataset[0]
    data = get_fixed_splits(data, aget(args, "dataset"), fold)
    data = data.to(aget(args, "device"))

    model = model_cls(data.edge_index, args).to(aget(args, "device"))

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {
            "params": sheaf_learner_params,
            "weight_decay": aget(args, "sheaf_decay"),
            "lr": aget(args, "maps_lr") if aget(args, "maps_lr") is not None else aget(args, "lr"),
        },
        {
            "params": other_params,
            "weight_decay": aget(args, "weight_decay"),
            "lr": aget(args, "lr"),
        }
    ])

    best_val_acc = 0.0
    best_val_loss = float("inf")
    test_acc = 0.0
    best_epoch = 0
    bad_counter = 0

    best_preds = None
    best_node_embeddings = None
    umap_reducer = None

    epochs = int(aget(args, "epochs", 200))
    early_stopping = int(aget(args, "early_stopping", 50))
    stop_strategy = str(aget(args, "stop_strategy", "acc"))
    task = str(aget(args, "task", "classification"))

    print(f"[info] Task: {task}, Fold: {fold}, Epochs: {epochs}")

    for epoch in range(epochs):
        train(model, optimizer, data, task=task)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data, task=task)

        if fold == 0:
            wandb_payload = {
                f"fold{fold}_train_loss": float(train_loss),
                f"fold{fold}_val_loss": float(val_loss),
                f"fold{fold}_tmp_test_loss": float(tmp_test_loss),
            }
            if task != "regression":
                wandb_payload.update({
                    f"fold{fold}_train_acc": float(train_acc),
                    f"fold{fold}_val_acc": float(val_acc),
                    f"fold{fold}_tmp_test_acc": float(tmp_test_acc),
                })
            else:
                train_pred = preds[0] if len(preds) > 0 else None
                val_pred = preds[1] if len(preds) > 1 else None
                test_pred = preds[2] if len(preds) > 2 else None
                train_target = data.y[data.train_mask] if hasattr(data, "train_mask") else data.y
                val_target = data.y[data.val_mask] if hasattr(data, "val_mask") else data.y
                test_target = data.y[data.test_mask] if hasattr(data, "test_mask") else data.y
                if train_pred is not None and train_pred.shape[1] >= 2 and train_target.shape[1] >= 2:
                    wandb_payload[f"fold{fold}_train_v_mag_mse"] = float(torch.mean((train_pred.cpu()[:,0] - train_target.cpu()[:,0]) ** 2).item())
                    wandb_payload[f"fold{fold}_train_theta_mse"] = float(torch.mean((train_pred.cpu()[:,-1] - train_target.cpu()[:,-1]) ** 2).item())
                if val_pred is not None and val_pred.shape[1] >= 2 and val_target.shape[1] >= 2:
                    wandb_payload[f"fold{fold}_val_v_mag_mse"] = float(torch.mean((val_pred.cpu()[:,0] - val_target.cpu()[:,0]) ** 2).item())
                    wandb_payload[f"fold{fold}_val_theta_mse"] = float(torch.mean((val_pred.cpu()[:,-1] - val_target.cpu()[:,-1]) ** 2).item())
                if test_pred is not None and test_pred.shape[1] >= 2 and test_target.shape[1] >= 2:
                    wandb_payload[f"fold{fold}_test_v_mag_mse"] = float(torch.mean((test_pred.cpu()[:,0] - test_target.cpu()[:,0]) ** 2).item())
                    wandb_payload[f"fold{fold}_test_theta_mse"] = float(torch.mean((test_pred.cpu()[:,-1] - test_target.cpu()[:,-1]) ** 2).item())
            wandb.log(wandb_payload, step=epoch)

        if task == "regression":
            new_best = val_loss < best_val_loss
        else:
            new_best = (val_acc > best_val_acc) if (stop_strategy == "acc") else (val_loss < best_val_loss)

        if new_best:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            test_acc = float(tmp_test_acc)
            best_epoch = int(epoch)
            bad_counter = 0
            if len(preds) > 2:
                best_preds = preds[2]
            try:
                model.eval()
                with torch.no_grad():
                    best_node_embeddings = model(data.x).detach().cpu()
            except Exception as e:
                print(f"[warn] Could not capture node embeddings at best epoch: {e}")
        else:
            bad_counter += 1

        if bad_counter == early_stopping:
            break

        if fold == 0 and epoch % 5 == 0:
            umap_reducer = save_umap_checkpoint(model=model, data=data, labels=data.y,
                                                fold=fold, epoch=epoch, umap_reducer=umap_reducer)

        print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")

    if best_preds is not None:
        save_predictions(best_preds, fold, tag="U_matrix_best")
    else:
        print(f"[warn] No predictions to save for fold {fold}.")

    if best_node_embeddings is not None:
        save_predictions(best_node_embeddings, fold, tag="node_embeddings_best")

    if len(preds) > 2:
        save_predictions(preds[2], fold, tag="U_matrix_final")

    save_model_artifacts(model, args, fold)

    try:
        labels_path = os.path.join(get_save_dir("outputs"), f"labels_fold{fold}.pt")
        torch.save(data.y.cpu(), labels_path)
        print(f"[save] Saved labels for fold {fold} to: {labels_path}")
    except Exception as e:
        print(f"[warn] Could not save labels for fold {fold}: {e}")

    import matplotlib.pyplot as plt
    import io

    if task == "regression":
        print(f"Test loss: {test_acc:.4f}")
        print(f"Best val loss: {best_val_loss:.4f}")
        if best_preds is not None:
            test_pred = best_preds
            test_target = data.y[data.test_mask].cpu() if hasattr(data, "test_mask") else data.y.cpu()
            if test_pred.shape[1] >= 2 and test_target.shape[1] >= 2:
                mse_dim = ((test_pred - test_target) ** 2).mean(dim=0).numpy()
                v_mag_mse = float(mse_dim[0])
                theta_mse = float(mse_dim[-1])
                wandb.log({
                    f"fold{fold}_final_test_mse_v_mag": v_mag_mse,
                    f"fold{fold}_final_test_mse_theta": theta_mse,
                })
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].plot(np.arange(len(test_pred)), test_pred[:,0], label='V_pred')
                ax[0].set_title('V Prediction')
                ax[0].set_xlabel('Node')
                ax[0].set_ylabel('V')
                ax[0].legend()
                ax[1].plot(np.arange(len(test_pred)), test_pred[:,-1], label='theta_pred', color='orange')
                ax[1].set_title('Theta Prediction')
                ax[1].set_xlabel('Node')
                ax[1].set_ylabel('Theta')
                ax[1].legend()
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                wandb.log({f"fold{fold}_V_theta_plot": wandb.Image(buf, caption="V and Theta Predictions")})
                plt.close(fig)
    else:
        print(f"Test acc: {test_acc:.4f}")
        print(f"Best val acc: {best_val_acc:.4f}")

    if "ODE" not in str(aget(args, "model", "")):
        try:
            for i in range(len(model.sheaf_learners)):
                L_max = model.sheaf_learners[i].L.detach().max().item()
                L_min = model.sheaf_learners[i].L.detach().min().item()
                L_avg = model.sheaf_learners[i].L.detach().mean().item()
                L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
                print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")
            with np.printoptions(precision=3, suppress=True):
                for i in range(0, int(aget(args, "layers", 0))):
                    print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")
        except Exception as e:
            print(f"[warn] discrete debug failed: {e}")

    if task == "regression":
        wandb.log({"best_test_loss": float(test_acc), "best_val_loss": float(best_val_loss), "best_epoch": int(best_epoch)})
    else:
        wandb.log({"best_test_acc": float(test_acc), "best_val_acc": float(best_val_acc), "best_epoch": int(best_epoch)})

    keep_running = float(test_acc) >= float(aget(args, "min_acc", 0.0)) if task != "regression" else True
    return float(test_acc), float(best_val_acc), keep_running


def run_exp_snapshot_classic(args, dataset, model_cls, fold: int) -> Tuple[float, float, bool]:
    data0 = dataset[0]
    data0 = data0.to(aget(args, "device"))

    model = model_cls(data0.edge_index, args).to(aget(args, "device"))

    device = aget(args, "device")
    edge_attr = data0.edge_attr.to(device) if data0.edge_attr is not None else None
    if edge_attr is not None:
        edge_feat_dim = edge_attr.shape[1]
        print(f"[edge_attr] shape: {edge_attr.shape} — using physical line features (G_ij, B_ij)")
    else:
        edge_feat_dim = 0
        print("[edge_attr] not found in dataset — sheaf learner will use node features only")

    # PolySpectralGNN has no sheaf learner — force edge_feat_dim=0 and don't pass edge_attr
    if aget(args, "model") == "PolySpectralGNN":
        edge_feat_dim = 0
        edge_attr = None
        print("[edge_attr] PolySpectralGNN baseline — edge features disabled")

    # Inject edge_feat_dim into args so the model constructor sizes linear layers correctly
    try:
        if isinstance(args, dict):
            args['edge_feat_dim'] = edge_feat_dim
        else:
            args.edge_feat_dim = edge_feat_dim
    except Exception:
        try:
            wandb.config.update({'edge_feat_dim': edge_feat_dim}, allow_val_change=True)
        except Exception:
            pass

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {
            "params": sheaf_learner_params,
            "weight_decay": aget(args, "sheaf_decay"),
            "lr": aget(args, "maps_lr") if aget(args, "maps_lr") is not None else aget(args, "lr"),
        },
        {
            "params": other_params,
            "weight_decay": aget(args, "weight_decay"),
            "lr": aget(args, "lr"),
        }
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    best_val_loss = float("inf")
    test_loss = float("inf")
    best_val_mae = float("inf")
    test_mae = float("inf")
    best_val_rmse = float("inf")
    test_rmse = float("inf")
    best_val_mse_per_dim = None
    test_mse_per_dim = None
    best_val_mae_per_dim = None
    test_mae_per_dim = None
    best_epoch = 0
    bad_counter = 0
    best_preds = None

    epochs = int(aget(args, "epochs", 200))
    early_stopping = int(aget(args, "early_stopping", 50))

    n_snapshots = len(dataset)
    train_idx, val_idx, test_idx = split_snapshot_indices(
        n_snapshots,
        int(aget(args, "seed", 0)) + int(fold),
        float(aget(args, "snapshot_train_ratio", 0.85)),
        float(aget(args, "snapshot_val_ratio", 0.05)),
    )
    train_samples_per_epoch = int(aget(args, "snapshot_train_samples_per_epoch", 0) or 0)
    eval_samples = int(aget(args, "snapshot_eval_samples", 0) or 0)
    batch_size = int(aget(args, "batch_size", 32))

    # Compute normalisation statistics from training set only (no data leakage).
    # For cross-grid datasets, use the first grid as placeholder — per-grid stats
    # are computed below and used in the actual training/eval loops.
    device = aget(args, "device")
    if hasattr(dataset, 'get_grid_datasets'):
        first_grid = dataset.get_grid_datasets()[0]
        n_train_first = int(0.85 * len(first_grid))
        x_mean, x_std, y_mean, y_std, const_x_mask = _compute_normalisation(first_grid, np.arange(n_train_first))
    else:
        x_mean, x_std, y_mean, y_std, const_x_mask = _compute_normalisation(dataset, train_idx)
    x_mean = x_mean.to(device)
    x_std  = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    # For cross-grid training, compute per-grid normalisation stats.
    # Each grid has different node counts so global stats cannot be stacked.
    # Single-grid runs skip this entirely (grid_norm_stats stays empty).
    grid_norm_stats = {}
    if hasattr(dataset, 'get_grid_datasets'):
        print("[cross-grid] Computing per-grid normalisation statistics...")
        for grid_ds in dataset.get_grid_datasets():
            n_grid = len(grid_ds)
            n_train_grid = int(0.85 * n_grid)
            grid_train_idx = np.arange(n_train_grid)
            gx_mean, gx_std, gy_mean, gy_std, _ = _compute_normalisation(grid_ds, grid_train_idx)
            grid_norm_stats[id(grid_ds)] = (
                gx_mean.to(device), gx_std.to(device),
                gy_mean.to(device), gy_std.to(device)
            )
            print(f"[cross-grid] Normalisation computed for grid with {grid_ds.edge_index.shape} edge_index")

    mse_dim_labels = ["p_gen", "q_gen", "v", "theta"]

    def _mse_dim_key(prefix: str, idx: int) -> str:
        label = mse_dim_labels[idx] if idx < len(mse_dim_labels) else f"dim{idx}"
        return f"{prefix}_{label}"

    def _load_batch(indices):
        data_list = [_normalise_data(dataset[int(i)].to(device), x_mean, x_std, y_mean, y_std)
                     for i in indices]
        x_batch = torch.stack([d.x for d in data_list], dim=0)
        y_batch = torch.stack([d.y for d in data_list], dim=0)
        if hasattr(data_list[0], 'loss_mask') and data_list[0].loss_mask is not None:
            mask_batch = torch.stack([d.loss_mask for d in data_list], dim=0)
        else:
            mask_batch = None
        return x_batch, y_batch, mask_batch

    for epoch in range(epochs):
        train_losses = []
        train_maes = []
        train_rmses = []
        train_mse_per_dim = []
        train_mae_per_dim = []

        if train_samples_per_epoch > 0 and len(train_idx) > train_samples_per_epoch:
            epoch_train_idx = np.random.choice(train_idx, size=train_samples_per_epoch, replace=False)
        else:
            epoch_train_idx = train_idx.copy()
        np.random.shuffle(epoch_train_idx)

        # Cross-grid-aware training batch loop.
        # If dataset is UnifiedPowerGridDataset, group each batch by grid,
        # call model.update_edge_index before each grid's sub-batch.
        # Single-grid runs use the original else branch.
        for batch_start in range(0, len(epoch_train_idx), batch_size):
            batch_idx = epoch_train_idx[batch_start: batch_start + batch_size]

            if hasattr(dataset, 'get_grid_for_index'):
                from collections import defaultdict
                grid_groups = defaultdict(list)
                for idx in batch_idx:
                    grid_ds, local_idx = dataset.get_grid_for_index(int(idx))
                    grid_groups[id(grid_ds)].append((grid_ds, local_idx))

                for grid_id, items in grid_groups.items():
                    grid_ds = items[0][0]
                    local_indices = [item[1] for item in items]

                    model.update_edge_index(grid_ds.edge_index.to(device))
                    grid_edge_attr = grid_ds.edge_attr.to(device) if grid_ds.edge_attr is not None else None

                    grid_norm = grid_norm_stats.get(id(grid_ds))
                    if grid_norm is None:
                        gx_mean, gx_std, gy_mean, gy_std = x_mean, x_std, y_mean, y_std
                    else:
                        gx_mean, gx_std, gy_mean, gy_std = grid_norm

                    data_list = [_normalise_data(grid_ds[i].to(device), gx_mean, gx_std, gy_mean, gy_std)
                                 for i in local_indices]
                    x_batch = torch.stack([d.x for d in data_list], dim=0)
                    y_batch = torch.stack([d.y for d in data_list], dim=0)
                    if hasattr(data_list[0], 'loss_mask') and data_list[0].loss_mask is not None:
                        mask_batch = torch.stack([d.loss_mask for d in data_list], dim=0)
                    else:
                        mask_batch = None

                    if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                        print(f"[warn] NaN in batch for grid {grid_ds.edge_index.shape} — skipping")
                        continue

                    loss, mae, rmse, mse_per_dim, mae_per_dim = train_snapshot_batch(
                        model, optimizer, x_batch, y_batch,
                        edge_attr=grid_edge_attr, mask_batch=mask_batch
                    )
                    train_losses.append(loss)
                    train_maes.append(mae)
                    train_rmses.append(rmse)
                    train_mse_per_dim.append(mse_per_dim)
                    train_mae_per_dim.append(mae_per_dim)

            else:
                # Original single-grid path — unchanged
                x_batch, y_batch, mask_batch = _load_batch(batch_idx)
                loss, mae, rmse, mse_per_dim, mae_per_dim = train_snapshot_batch(
                    model, optimizer, x_batch, y_batch, edge_attr=edge_attr, mask_batch=mask_batch
                )
                train_losses.append(loss)
                train_maes.append(mae)
                train_rmses.append(rmse)
                train_mse_per_dim.append(mse_per_dim)
                train_mae_per_dim.append(mae_per_dim)

        if eval_samples > 0 and len(val_idx) > eval_samples:
            eval_val_idx = np.random.choice(val_idx, size=eval_samples, replace=False)
        else:
            eval_val_idx = val_idx
        if eval_samples > 0 and len(test_idx) > eval_samples:
            eval_test_idx = np.random.choice(test_idx, size=eval_samples, replace=False)
        else:
            eval_test_idx = test_idx

        # Cross-grid-aware eval loop, mirroring the train loop.
        def _eval_batched(idx_list):
            metrics = []
            for batch_start in range(0, len(idx_list), batch_size):
                batch_idx = idx_list[batch_start: batch_start + batch_size]

                if hasattr(dataset, 'get_grid_for_index'):
                    from collections import defaultdict
                    grid_groups = defaultdict(list)
                    for idx in batch_idx:
                        grid_ds, local_idx = dataset.get_grid_for_index(int(idx))
                        grid_groups[id(grid_ds)].append((grid_ds, local_idx))

                    for grid_id, items in grid_groups.items():
                        grid_ds = items[0][0]
                        local_indices = [item[1] for item in items]
                        model.update_edge_index(grid_ds.edge_index.to(device))
                        grid_edge_attr = grid_ds.edge_attr.to(device) if grid_ds.edge_attr is not None else None
                        grid_norm = grid_norm_stats.get(id(grid_ds))
                        if grid_norm is None:
                            gx_mean, gx_std, gy_mean, gy_std = x_mean, x_std, y_mean, y_std
                        else:
                            gx_mean, gx_std, gy_mean, gy_std = grid_norm
                        data_list = [_normalise_data(grid_ds[i].to(device), gx_mean, gx_std, gy_mean, gy_std)
                                     for i in local_indices]
                        x_batch = torch.stack([d.x for d in data_list], dim=0)
                        y_batch = torch.stack([d.y for d in data_list], dim=0)
                        if hasattr(data_list[0], 'loss_mask') and data_list[0].loss_mask is not None:
                            mask_batch = torch.stack([d.loss_mask for d in data_list], dim=0)
                        else:
                            mask_batch = None
                        metrics.append(eval_snapshot_batch(model, x_batch, y_batch,
                                                           edge_attr=grid_edge_attr, mask_batch=mask_batch))
                else:
                    x_batch, y_batch, mask_batch = _load_batch(batch_idx)
                    metrics.append(eval_snapshot_batch(model, x_batch, y_batch,
                                                       edge_attr=edge_attr, mask_batch=mask_batch))
            return metrics

        val_metrics  = _eval_batched(eval_val_idx)
        test_metrics = _eval_batched(eval_test_idx)

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean([m[0] for m in val_metrics])) if val_metrics else float("inf")
        tmp_test_loss = float(np.mean([m[0] for m in test_metrics])) if test_metrics else float("inf")
        train_mae = float(np.mean(train_maes)) if train_maes else float("inf")
        val_mae = float(np.mean([m[1] for m in val_metrics])) if val_metrics else float("inf")
        tmp_test_mae = float(np.mean([m[1] for m in test_metrics])) if test_metrics else float("inf")
        train_rmse = float(np.mean(train_rmses)) if train_rmses else float("inf")
        val_rmse = float(np.mean([m[2] for m in val_metrics])) if val_metrics else float("inf")
        tmp_test_rmse = float(np.mean([m[2] for m in test_metrics])) if test_metrics else float("inf")
        train_mse_dim = np.mean(np.stack(train_mse_per_dim, axis=0), axis=0) if train_mse_per_dim else None
        val_mse_dim = np.mean(np.stack([m[3] for m in val_metrics], axis=0), axis=0) if val_metrics else None
        test_mse_dim = np.mean(np.stack([m[3] for m in test_metrics], axis=0), axis=0) if test_metrics else None
        train_mae_dim = np.mean(np.stack(train_mae_per_dim, axis=0), axis=0) if train_mae_per_dim else None
        val_mae_dim = np.mean(np.stack([m[4] for m in val_metrics], axis=0), axis=0) if val_metrics else None
        test_mae_dim = np.mean(np.stack([m[4] for m in test_metrics], axis=0), axis=0) if test_metrics else None

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mae={val_mae:.4f} | lr={current_lr:.2e}",
            flush=True,
        )

        if fold == 0:
            wandb_payload = {
                f"fold{fold}_train_loss": float(train_loss),
                f"fold{fold}_val_loss": float(val_loss),
                f"fold{fold}_tmp_test_loss": float(tmp_test_loss),
                f"fold{fold}_train_mae": float(train_mae),
                f"fold{fold}_val_mae": float(val_mae),
                f"fold{fold}_tmp_test_mae": float(tmp_test_mae),
                f"fold{fold}_train_rmse": float(train_rmse),
                f"fold{fold}_val_rmse": float(val_rmse),
                f"fold{fold}_tmp_test_rmse": float(tmp_test_rmse),
                f"fold{fold}_lr": float(current_lr),
            }
            if train_mse_dim is not None:
                for i, val in enumerate(train_mse_dim):
                    wandb_payload[_mse_dim_key(f"fold{fold}_train_mse", i)] = float(val)
            if val_mse_dim is not None:
                for i, val in enumerate(val_mse_dim):
                    wandb_payload[_mse_dim_key(f"fold{fold}_val_mse", i)] = float(val)
            if test_mse_dim is not None:
                for i, val in enumerate(test_mse_dim):
                    wandb_payload[_mse_dim_key(f"fold{fold}_tmp_test_mse", i)] = float(val)
            if train_mae_dim is not None:
                for i, val in enumerate(train_mae_dim):
                    wandb_payload[_mse_dim_key(f"fold{fold}_train_mae", i)] = float(val)
            if val_mae_dim is not None:
                for i, val in enumerate(val_mae_dim):
                    wandb_payload[_mse_dim_key(f"fold{fold}_val_mae", i)] = float(val)
            if test_mae_dim is not None:
                for i, val in enumerate(test_mae_dim):
                    wandb_payload[_mse_dim_key(f"fold{fold}_tmp_test_mae", i)] = float(val)
            wandb_payload.update({
                f"fold{fold}_train_acc": float("nan"),
                f"fold{fold}_val_acc": float("nan"),
                f"fold{fold}_tmp_test_acc": float("nan"),
            })
            wandb.log(wandb_payload, step=epoch)

        if fold == 0 and best_preds is not None and best_preds.shape[1] >= 2:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import io
                test_pred = best_preds.detach().cpu().float().numpy()
                test_target = data0.y.detach().cpu().float().numpy()
                if test_target.shape[1] >= 2:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].scatter(test_target[:, 0], test_pred[:, 0], alpha=0.4, s=8)
                    axes[0].plot([test_target[:, 0].min(), test_target[:, 0].max()],
                                 [test_target[:, 0].min(), test_target[:, 0].max()], 'r--', linewidth=1)
                    axes[0].set_title(f'V Magnitude | Epoch {epoch}')
                    axes[0].set_xlabel('Ground Truth')
                    axes[0].set_ylabel('Predicted')
                    axes[1].scatter(test_target[:, -1], test_pred[:, -1], alpha=0.4, s=8, color='orange')
                    axes[1].plot([test_target[:, -1].min(), test_target[:, -1].max()],
                                 [test_target[:, -1].min(), test_target[:, -1].max()], 'r--', linewidth=1)
                    axes[1].set_title(f'Theta | Epoch {epoch}')
                    axes[1].set_xlabel('Ground Truth')
                    axes[1].set_ylabel('Predicted')
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    wandb.log({f"fold{fold}_V_theta_scatter": wandb.Image(buf, caption=f"Epoch {epoch}")}, step=epoch)
                    plt.close(fig)
            except Exception as e:
                print(f"[warn] V/theta scatter plot failed for epoch {epoch}: {e}")

        if val_mae < best_val_mae:
            best_val_loss = float(val_loss)
            test_loss = float(tmp_test_loss)
            best_val_mae = float(val_mae)
            test_mae = float(tmp_test_mae)
            best_val_rmse = float(val_rmse)
            test_rmse = float(tmp_test_rmse)
            best_val_mse_per_dim = val_mse_dim
            test_mse_per_dim = test_mse_dim
            best_val_mae_per_dim = val_mae_dim
            test_mae_per_dim = test_mae_dim
            best_epoch = int(epoch)
            bad_counter = 0
            try:
                model.eval()
                with torch.no_grad():
                    norm_data0 = _normalise_data(data0, x_mean, x_std, y_mean, y_std)
                    best_preds = _denormalise_preds(model(norm_data0.x, edge_attr=edge_attr).detach().cpu(), y_mean, y_std)
            except Exception as e:
                print(f"[warn] Could not capture predictions at best epoch (snapshot): {e}")
        else:
            bad_counter += 1

        if bad_counter == early_stopping:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.4f} | Best val loss: {best_val_loss:.4f}")
    print(f"Test MAE:  {test_mae:.4f} | Best val MAE:  {best_val_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} | Best val RMSE: {best_val_rmse:.4f}")

    run_dir = _get_run_dir()

    if best_preds is not None:
        save_path = run_dir / f"U_matrix_snapshot_best_fold{fold}.pt"
        torch.save(best_preds, save_path)
        print(f"[save] Saved U_matrix_snapshot_best for fold {fold} to: {save_path}")
    else:
        print(f"[warn] No snapshot predictions to save for fold {fold}.")

    try:
        labels_path = run_dir / f"labels_snapshot_fold{fold}.pt"
        torch.save(data0.y.cpu(), labels_path)
        print(f"[save] Saved snapshot labels for fold {fold} to: {labels_path}")
    except Exception as e:
        print(f"[warn] Could not save snapshot labels for fold {fold}: {e}")

    save_model_artifacts(model, args, fold)

    wandb.log({
        "best_test_loss": float(test_loss),
        "best_val_loss": float(best_val_loss),
        "best_test_mae": float(test_mae),
        "best_val_mae": float(best_val_mae),
        "best_test_rmse": float(test_rmse),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "best_test_acc": float("nan"),
        "best_val_acc": float("nan"),
    })
    if best_val_mse_per_dim is not None:
        wandb.log({_mse_dim_key("best_val_mse", i): float(v) for i, v in enumerate(best_val_mse_per_dim)})
    if test_mse_per_dim is not None:
        wandb.log({_mse_dim_key("best_test_mse", i): float(v) for i, v in enumerate(test_mse_per_dim)})
    if best_val_mae_per_dim is not None:
        wandb.log({_mse_dim_key("best_val_mae", i): float(v) for i, v in enumerate(best_val_mae_per_dim)})
    if test_mae_per_dim is not None:
        wandb.log({_mse_dim_key("best_test_mae", i): float(v) for i, v in enumerate(test_mae_per_dim)})

    return float(test_loss), float(best_val_loss), True


# =====================================================================
#  RESOURCE-ANALYSIS FOLD RUN
# =====================================================================
def run_exp_resource(args, dataset, model_cls, fold: int) -> Tuple[float, float, bool, Dict[str, Any]]:
    base_seed = int(aget(args, "seed", 0))
    fseed = fold_seed(base_seed, fold)

    deterministic = bool(aget(args, "deterministic", True))
    strict = truthy(os.environ.get("STRICT_DETERMINISM", "0")) or bool(aget(args, "strict_determinism", False))
    set_reproducible(fseed, deterministic=deterministic, strict=strict)

    data = dataset[0]
    data = get_fixed_splits(data, aget(args, "dataset"), fold)

    device = normalize_device(aget(args, "device", "cpu"))
    data = data.to(device)

    max_epochs = int(aget(args, "epochs", 0))
    fold_step_base = int(fold) * int(max_epochs)

    best_preds = None
    best_node_embeddings = None
    umap_reducer = None

    mon = None
    try:
        cuda_idx = device_cuda_index(device)
        log_every_s = float(aget(args, "sys_log_every_s", 1.0))
        mon = ResourceMonitor(
            cuda_index=cuda_idx,
            log_every_s=log_every_s,
            disk_path=".",
            prefix=f"fold{fold}_sys",
        )
        mon.start()

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        model = model_cls(data.edge_index, args).to(device)

        macs = maybe_profile_macs_torchprofile(model, data.x)
        if macs is not None:
            wandb.log({
                "global_step": fold_step_base,
                "fold": int(fold),
                "epoch": 0,
                f"fold{fold}_macs_forward_torchprofile": float(macs),
                f"fold{fold}_flops_forward_from_macs": float(2.0 * macs),
            })

        sheaf_learner_params, other_params = model.grouped_parameters()
        optimizer = torch.optim.Adam([
            {
                "params": sheaf_learner_params,
                "weight_decay": float(aget(args, "sheaf_decay")),
                "lr": float(aget(args, "maps_lr")) if aget(args, "maps_lr") is not None else float(aget(args, "lr")),
            },
            {
                "params": other_params,
                "weight_decay": float(aget(args, "weight_decay")),
                "lr": float(aget(args, "lr")),
            }
        ])

        best_val_acc = 0.0
        best_val_loss = float("inf")
        test_acc = 0.0
        best_epoch = 0
        bad_counter = 0

        epochs = int(aget(args, "epochs", 200))
        early_stopping = int(aget(args, "early_stopping", 50))
        stop_strategy = str(aget(args, "stop_strategy", "acc")).lower()

        t_fold0 = time.perf_counter()
        t_best = None
        val_acc_hist: List[float] = []
        val_loss_hist: List[float] = []
        step_times: List[float] = []
        flops_samples: List[float] = []

        profile_flops_flag = bool(aget(args, "profile_flops", True))
        profile_flops = profile_flops_flag and profiler_available()
        flops_profile_epochs = int(aget(args, "flops_profile_epochs", 1))

        for epoch in range(epochs):
            global_step = fold_step_base + int(epoch)

            do_profile_now = profile_flops and (epoch < flops_profile_epochs)
            flops, step_time_s = train_step_with_optional_flops(
                enabled=profile_flops,
                device=device,
                do_profile_now=do_profile_now,
                train_fn=train,
                model=model,
                optimizer=optimizer,
                data=data,
            )
            step_times.append(float(step_time_s))
            if flops is not None:
                flops_samples.append(float(flops))

            [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)

            if fold == 0 and len(preds) > 2 and preds[2].shape[1] >= 2:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import io
                    test_pred = preds[2].detach().cpu().float().numpy()
                    test_target = (data.y[data.test_mask] if hasattr(data, "test_mask") else data.y).detach().cpu().float().numpy()
                    if test_target.shape[1] >= 2:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].scatter(test_target[:, 0], test_pred[:, 0], alpha=0.4, s=8)
                        axes[0].plot([test_target[:, 0].min(), test_target[:, 0].max()],
                                     [test_target[:, 0].min(), test_target[:, 0].max()], 'r--', linewidth=1)
                        axes[0].set_title(f'V Magnitude | Epoch {epoch}')
                        axes[0].set_xlabel('Ground Truth')
                        axes[0].set_ylabel('Predicted')
                        axes[1].scatter(test_target[:, -1], test_pred[:, -1], alpha=0.4, s=8, color='orange')
                        axes[1].plot([test_target[:, -1].min(), test_target[:, -1].max()],
                                     [test_target[:, -1].min(), test_target[:, -1].max()], 'r--', linewidth=1)
                        axes[1].set_title(f'Theta | Epoch {epoch}')
                        axes[1].set_xlabel('Ground Truth')
                        axes[1].set_ylabel('Predicted')
                        plt.tight_layout()
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        wandb.log({f"fold{fold}_V_theta_scatter": wandb.Image(buf, caption=f"Epoch {epoch}")}, step=epoch)
                        plt.close(fig)
                except Exception as e:
                    print(f"[warn] V/theta scatter plot failed for epoch {epoch}: {e}")

            if fold == 0:
                wandb.log({
                    "global_step": global_step,
                    "fold": int(fold),
                    "epoch": int(epoch),
                    f"fold{fold}_train_acc": float(train_acc),
                    f"fold{fold}_train_loss": float(train_loss),
                    f"fold{fold}_val_acc": float(val_acc),
                    f"fold{fold}_val_loss": float(val_loss),
                    f"fold{fold}_tmp_test_acc": float(tmp_test_acc),
                    f"fold{fold}_tmp_test_loss": float(tmp_test_loss),
                })

            val_acc_hist.append(float(val_acc))
            val_loss_hist.append(float(val_loss))

            alloc_gb = reserv_gb = max_alloc_gb = max_reserv_gb = None
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                try:
                    torch.cuda.synchronize()
                    alloc_gb = torch.cuda.memory_allocated() / 1e9
                    reserv_gb = torch.cuda.memory_reserved() / 1e9
                    max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
                    max_reserv_gb = torch.cuda.max_memory_reserved() / 1e9
                except Exception:
                    pass

            wandb.log({
                "global_step": global_step,
                "fold": int(fold),
                "epoch": int(epoch),
                f"fold{fold}_time_epoch_s": float(step_time_s) if step_time_s is not None else None,
                f"fold{fold}_flops_epoch_profiler": float(flops) if flops is not None else None,
                f"fold{fold}_torch_mem_alloc_gb": alloc_gb,
                f"fold{fold}_torch_mem_reserved_gb": reserv_gb,
                f"fold{fold}_torch_max_mem_alloc_gb": max_alloc_gb,
                f"fold{fold}_torch_max_mem_reserved_gb": max_reserv_gb,
            })

            improved = (val_acc > best_val_acc) if (stop_strategy == "acc") else (val_loss < best_val_loss)
            if improved:
                best_val_acc = float(val_acc)
                best_val_loss = float(val_loss)
                test_acc = float(tmp_test_acc)
                best_epoch = int(epoch)
                bad_counter = 0
                t_best = time.perf_counter()
                if len(preds) > 2:
                    best_preds = preds[2]
                try:
                    model.eval()
                    with torch.no_grad():
                        best_node_embeddings = model(data.x).detach().cpu()
                except Exception as e:
                    print(f"[warn] Could not capture node embeddings at best epoch (resource): {e}")
            else:
                bad_counter += 1

            if bad_counter >= early_stopping:
                break

            if fold == 0 and epoch % 5 == 0:
                umap_reducer = save_umap_checkpoint(model=model, data=data, labels=data.y,
                                                    fold=fold, epoch=epoch, umap_reducer=umap_reducer)

        print(f"Fold {fold} | Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f}")

        if best_preds is not None:
            save_predictions(best_preds, fold, tag="U_matrix_resource_best")
        else:
            print(f"[warn] No resource predictions to save for fold {fold}. preds length: {len(preds)}")

        if best_node_embeddings is not None:
            save_predictions(best_node_embeddings, fold, tag="node_embeddings_resource_best")

        if len(preds) > 2:
            save_predictions(preds[2], fold, tag="U_matrix_resource_final")

        save_model_artifacts(model, args, fold)

        try:
            labels_path = os.path.join(get_save_dir("outputs"), f"labels_fold{fold}.pt")
            torch.save(data.y.cpu(), labels_path)
            print(f"[save] Saved labels for fold {fold} to: {labels_path}")
        except Exception as e:
            print(f"[warn] Could not save labels for fold {fold}: {e}")

        fold_summary: Dict[str, Any] = {
            "fold": int(fold),
            "fold_seed": int(fseed),
            f"fold{fold}_best_test_acc": float(test_acc),
            f"fold{fold}_best_val_acc": float(best_val_acc),
            f"fold{fold}_best_epoch": int(best_epoch),
            "deterministic_enabled": bool(deterministic),
            "strict_determinism_enabled": bool(strict),
        }

        fold_time_s = time.perf_counter() - t_fold0
        sys_agg = mon.aggregates() if mon is not None else {}

        avg_step_time_s = float(np.mean(step_times)) if step_times else None
        avg_step_time_ms = (1000.0 * avg_step_time_s) if avg_step_time_s is not None else None

        avg_flops_per_epoch = float(np.mean(flops_samples)) if flops_samples else None
        avg_gflops_per_epoch = (avg_flops_per_epoch / 1e9) if avg_flops_per_epoch is not None else None

        fold_summary.update({
            f"fold{fold}_fold_time_s": float(fold_time_s),
            f"fold{fold}_time_to_best_s": float((t_best - t_fold0) if t_best is not None else fold_time_s),
            f"fold{fold}_avg_step_time_ms": avg_step_time_ms,
            f"fold{fold}_avg_flops_per_epoch_profiler": avg_flops_per_epoch,
            f"fold{fold}_avg_gflops_per_epoch_profiler": avg_gflops_per_epoch,
            **sys_agg,
        })

        wandb.log(fold_summary)

        keep_running = float(test_acc) >= float(aget(args, "min_acc", 0.0))
        return float(test_acc), float(best_val_acc), keep_running, fold_summary

    finally:
        if mon is not None:
            try:
                mon.stop()
            except Exception:
                pass


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except Exception:
        sha = "unknown"

    # ---------------- Model routing ----------------
    if args.model == "DiagSheafODE":
        model_cls = DiagSheafDiffusion
    elif args.model == "BundleSheafODE":
        model_cls = BundleSheafDiffusion
    elif args.model == "GeneralSheafODE":
        model_cls = GeneralSheafDiffusion
    elif args.model in ("DiagSheafODEPolynomial", "DiagSheafODEPoly"):
        model_cls = DiagSheafDiffusion_Polynomial
    elif args.model in ("BundleSheafODEPolynomial", "BundleSheafODEPoly"):
        model_cls = BundleSheafDiffusion_Polynomial
    elif args.model in ("GeneralSheafODEPolynomial", "GeneralSheafODEPoly"):
        model_cls = GeneralSheafDiffusion_Polynomial
    elif args.model == "DiagSheaf":
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == "BundleSheaf":
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == "GeneralSheaf":
        model_cls = DiscreteGeneralSheafDiffusion
    elif args.model == "DiagSheafPolynomial":
        model_cls = DiscreteDiagSheafDiffusionPolynomial
    elif args.model == "BundleSheafPolynomial":
        model_cls = DiscreteBundleSheafDiffusionPolynomial
    elif args.model == "GeneralSheafPolynomial":
        model_cls = DiscreteGeneralSheafDiffusionPolynomial
    elif args.model == "PolySpectralGNN":
        model_cls = PolySpectralGNN
    else:
        raise ValueError(f"Unknown model {args.model}")

    # ---------------- Dataset ----------------
    if args.dataset == "synthetic_exp":
        dataset = get_synthetic_dataset(args.dataset, args)
    else:
        dataset = get_dataset(args.dataset)

    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # ---------------- Enrich args ----------------
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset[0].x.shape[1]
    if is_snapshot_dataset(dataset) and str(getattr(args, "task", "classification")) != "regression":
        args.task = "regression"

    if str(getattr(args, "task", "classification")) == "regression":
        y0 = dataset[0].y
        args.output_dim = int(y0.shape[1]) if y0.dim() > 1 else 1
    else:
        try:
            args.output_dim = dataset.num_classes
        except Exception:
            args.output_dim = torch.unique(dataset[0].y).shape[0]
    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    print(f"[info] Outputs will be saved to: {get_save_dir('outputs')}")

    # ---------------- mode switch ----------------
    resource_analysis = bool(getattr(args, "resource_analysis", False))

    if not resource_analysis:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        strict = truthy(os.environ.get("STRICT_DETERMINISM", "0")) or bool(getattr(args, "strict_determinism", False))
        set_reproducible(int(args.seed), deterministic=bool(getattr(args, "deterministic", True)), strict=bool(strict))

    print(f"Running with wandb account: {args.entity}")
    print(args)

    default_project = "Chameleon_BestResults_PolySD_vs_Standard" if not resource_analysis else "Convergence_Ablation_Chameleon"
    project_name = getattr(args, "wandb_project", None) or default_project

    wandb.init(
        project=project_name,
        entity=args.entity,
        config={**vars(args), "sha": sha},
        name=f"{args.model}-{args.dataset}-seed{args.seed}",
    )

    if resource_analysis:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

        try:
            wandb.config.update({
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cudnn_version": torch.backends.cudnn.version(),
                "resource_analysis": True,
                "profiler_available": bool(profiler_available()),
            }, allow_val_change=True)
        except Exception:
            pass

    results: List[List[float]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold in tqdm(range(int(args.folds))):
        if not resource_analysis:
            test_acc, best_val_acc, keep_running = run_exp_classic(wandb.config, dataset, model_cls, fold)
            results.append([test_acc, best_val_acc])
            if not keep_running:
                break
        else:
            test_acc, best_val_acc, keep_running, fold_summary = run_exp_resource(wandb.config, dataset, model_cls, fold)
            results.append([test_acc, best_val_acc])
            fold_summaries.append(fold_summary)
            if not keep_running:
                break

    # ---------------- aggregate across folds ----------------
    if str(getattr(args, "task", "classification")) == "regression":
        test_loss_mean, val_loss_mean = np.mean(results, axis=0)
        test_loss_std = np.sqrt(np.var(results, axis=0)[0])
        wandb_results = {
            "test_loss": float(test_loss_mean),
            "val_loss": float(val_loss_mean),
            "test_loss_std": float(test_loss_std),
        }
        if is_snapshot_dataset(dataset):
            wandb_results.update({
                "test_acc": float("nan"),
                "val_acc": float("nan"),
                "test_acc_std": float("nan"),
            })
        wandb.log(wandb_results)
    else:
        test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
        wandb_results = {"test_acc": float(test_acc_mean), "val_acc": float(val_acc_mean), "test_acc_std": float(test_acc_std)}
        wandb.log(wandb_results)

    if resource_analysis:
        gflops_vals = []
        time_ms_vals = []
        for fs in fold_summaries:
            for k, v in fs.items():
                if k.endswith("_avg_gflops_per_epoch_profiler") and v is not None:
                    gflops_vals.append(float(v))
                if k.endswith("_avg_step_time_ms") and v is not None:
                    time_ms_vals.append(float(v))

        avg_gflops_10fold = float(np.mean(gflops_vals)) if len(gflops_vals) else None
        std_gflops_10fold = float(np.std(gflops_vals)) if len(gflops_vals) else None
        avg_time_ms_10fold = float(np.mean(time_ms_vals)) if len(time_ms_vals) else None
        std_time_ms_10fold = float(np.std(time_ms_vals)) if len(time_ms_vals) else None

        wandb.log({
            "cv/avg_gflops_per_epoch_profiler_mean": avg_gflops_10fold,
            "cv/avg_gflops_per_epoch_profiler_std": std_gflops_10fold,
            "cv/avg_step_time_ms_mean": avg_time_ms_10fold,
            "cv/avg_step_time_ms_std": std_time_ms_10fold,
        })

        wandb.run.summary["cv/avg_gflops_per_epoch_profiler_mean"] = avg_gflops_10fold
        wandb.run.summary["cv/avg_gflops_per_epoch_profiler_std"] = std_gflops_10fold
        wandb.run.summary["cv/avg_step_time_ms_mean"] = avg_time_ms_10fold
        wandb.run.summary["cv/avg_step_time_ms_std"] = std_time_ms_10fold

    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f"{model_name} on {args.dataset} | SHA: {sha}")
    if str(getattr(args, "task", "classification")) == "regression":
        print(f"Test loss: {test_loss_mean:.4f} +/- {test_loss_std:.4f} | Val loss: {val_loss_mean:.4f}")
    else:
        print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}")


    print(f"[info] All outputs saved to: {get_save_dir('outputs')}")
