#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Data

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import (
    DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion, 
    DiagSheafDiffusion_Polynomial, BundleSheafDiffusion_Polynomial, GeneralSheafDiffusion_Polynomial
)
from models.disc_models import (
    DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion,
    DiscreteDiagSheafDiffusionPolynomial, DiscreteBundleSheafDiffusionPolynomial, DiscreteGeneralSheafDiffusionPolynomial
)
from utils.heterophilic import get_dataset, get_synthetic_dataset, get_fixed_splits
from source.graph import graph_to_sheaf_inputs, load_graph


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data):
    """One optimisation step on the current fold."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()
    optimizer.step()
    del out


def test(model, data):
    """
        Evaluate the model on the cached train/val/test masks.
    """
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def generate_splits(data, train=0.6, val=0.2, seed=None):
    """
        Construct random boolean masks (train/val/test) for a latent graph run.
    """
    n = data.x.shape[0]
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        masks = torch.randperm(n, generator=generator)
    else:
        masks = torch.randperm(n)
    train_index = int(train * n)
    val_index = train_index + int(val * n)
    val_index = min(val_index, n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[masks[0:train_index]] = True
    val_mask[masks[train_index:val_index]] = True
    test_mask[masks[val_index:]] = True
    return Data(x=data.x, edge_index=data.edge_index, y=data.y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def _resolve_arg(args, key, default=None):
    """
        Safely extract an attribute/entry from argparse.Namespace, wandb Config, or dict.
    """
    if isinstance(args, dict):
        return args.get(key, default)
    if hasattr(args, key):
        return getattr(args, key)
    try:
        return args[key]
    except Exception:
        return default


def _collect_restriction_maps(model):
    """
        Return a dict with the stored Laplacian/restriction maps for every sheaf learner.
    """
    maps = {}
    sheaf_learners = getattr(model, 'sheaf_learners', None)
    if sheaf_learners is None:
        return maps
    for idx, learner in enumerate(sheaf_learners):
        layer_maps = getattr(learner, 'L', None)
        if layer_maps is not None:
            maps[f'layer_{idx}'] = layer_maps.detach().cpu()
    return maps


def _checkpoint_directory(dataset_name, model_name):
    """
        Create the per-run checkpoint directory and return the filesystem Path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / 'source-causal' / 'model_checkpoint'
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = dataset_dir / f'{model_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_model_artifacts(model, args, fold):
    """
        Persist the full model state dict (on CPU) plus the learned restriction maps.
    """
    dataset_name = _resolve_arg(args, 'dataset', 'unknown')
    model_name = _resolve_arg(args, 'model', 'sheaf')
    dataset_name = str(dataset_name).lower().replace(' ', '_')
    model_name = str(model_name)
    try:
        run_dir = _checkpoint_directory(dataset_name, model_name)
        state_dict = OrderedDict(
            (k, v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in model.state_dict().items()
        )
        checkpoint = {
            'state_dict': state_dict,
            'metadata': {
                'dataset': dataset_name,
                'model': model_name,
                'fold': fold,
                'saved_at': datetime.now().isoformat()
            }
        }
        torch.save(checkpoint, run_dir / 'model.pt')
        restriction_maps = _collect_restriction_maps(model)
        torch.save(restriction_maps, run_dir / 'restriction_maps.pt')
        print(f"Saved model checkpoint and restriction maps to {run_dir}")
    except Exception as exc:
        print(f"Failed to save model artifacts: {exc}")


def select_device(preferred_index: int):
    """
    Try to honor the requested CUDA index, but gracefully fall back to CPU if the
    GPU is unavailable, incompatible, or out of range.
    """
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if 0 <= preferred_index < cuda_count:
            device = torch.device(f'cuda:{preferred_index}')
            try:
                _ = torch.zeros(0, device=device)
                return device
            except RuntimeError as err:
                print(f"CUDA device {preferred_index} unavailable ({err}). Falling back to CPU.")
        else:
            print(f"Requested CUDA device {preferred_index} but only {cuda_count} present. Using CPU.")
    else:
        print("CUDA not available. Running on CPU.")
    return torch.device('cpu')


def run_exp(args, dataset, model_cls, fold):
    """
        Train/evaluate a single fold and return the metrics.
    """
    data = dataset[0]
    if args['latent_graph_path']:
        split_seed = args['seed'] + fold
        data = generate_splits(
            data,
            train=args['latent_train_frac'],
            val=args['latent_val_frac'],
            seed=split_seed,
        )
    else:
        data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args).to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay'],
         'lr': args['maps_lr'] if args['maps_lr'] is not None else args['lr']},
        {'params': other_params, 'weight_decay': args['weight_decay'], 'lr': args['lr']}
    ])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        # ----- Core optimisation step -----
        train(model, optimizer, data)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)

        if fold == 0:
            # Only log per-epoch curves on fold 0 to reduce W&B noise.
            wandb.log({
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }, step=epoch)

        # Switch between accuracy- or loss-based early stopping.
        new_best = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    if "ODE" not in args['model']:
        # Debugging for discrete models
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args['layers']):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    # Persist best scores for this fold and decide whether to continue.
    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = test_acc >= args['min_acc']
    # Refresh the restriction maps before saving the artifacts.
    test(model, data)
    save_model_artifacts(model, args, fold)
    return test_acc, best_val_acc, keep_running


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # ---------------- Model routing ----------------
    
    ######################### Continuous Models #########################
    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args.model == 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    # --- ODE + Polynomial (continuous PolySD) ---
    elif args.model in ('DiagSheafODEPolynomial', 'DiagSheafODEPoly'):
        model_cls = DiagSheafDiffusion_Polynomial
    elif args.model in ('BundleSheafODEPolynomial', 'BundleSheafODEPoly'):
        model_cls = BundleSheafDiffusion_Polynomial
    elif args.model in ('GeneralSheafODEPolynomial', 'GeneralSheafODEPoly'):
        model_cls = GeneralSheafDiffusion_Polynomial
    ######################### Discrete Models #########################
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    # --- Discrete Polynomial (discrete PolySD) ---
    elif args.model == 'DiagSheafPolynomial':
        model_cls = DiscreteDiagSheafDiffusionPolynomial
    elif args.model == 'BundleSheafPolynomial':
        model_cls = DiscreteBundleSheafDiffusionPolynomial
    elif args.model == 'GeneralSheafPolynomial':
        model_cls = DiscreteGeneralSheafDiffusionPolynomial
    else:
        raise ValueError(f'Unknown model {args.model}')

    # Decide which dataset loader to use: latent graph, synthetic, or benchmark.
    if args.latent_graph_path:
        graph = load_graph(args.latent_graph_path)
        edge_index, node_features, node_labels = graph_to_sheaf_inputs(graph)
        if node_labels is None:
            raise ValueError("Loaded graph does not contain node labels.")
        dataset = [Data(x=node_features, edge_index=edge_index, y=node_labels)]
        args.dataset = "latent_graph"
    elif args.dataset == "synthetic_exp":
        dataset = get_synthetic_dataset(args.dataset, args)
    else:
        dataset = get_dataset(args.dataset)

    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments used downstream by the models/optimiser.
    args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset[0].x.shape[1]
    try:
        args.output_dim = dataset.num_classes
    except:
        args.output_dim = torch.unique(dataset[0].y).shape[0]
    args.device = select_device(args.cuda)
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything so folds are reproducible.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)

    # Start a single W&B run; folds stream metrics into the same run.
    # Allow offline/debug execution when no entity/project is configured
    wandb_settings = {"project": "SplitSheafLearning", "config": vars(args)}
    if args.entity:
        wandb_settings["entity"] = args.entity
    else:
        os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(**wandb_settings)

    for fold in tqdm(range(args.folds)):
        # Each fold reuses the pre-loaded dataset but different masks.
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    # Aggregate the metrics converted to percentages for readability.
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')