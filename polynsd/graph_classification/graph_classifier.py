#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import functools
from typing import Literal, NamedTuple, Callable

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch_geometric.data import Data
from torch_geometric.nn import Linear, global_mean_pool, global_max_pool, global_add_pool
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, F1Score
from polynsd.graph_classification.pooling import AttentionPooling

class CommonStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor


class GraphClassifier(L.LightningModule):
    """Graph-level classification for standard GNNs (non-sheaf)."""

    def __init__(
        self,
        model: nn.Module,
        hidden_channels: int = 256,
        out_channels: int = 2,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        pooling: Literal["mean", "max", "add", "attention"] = "mean",
        sheaf_model: bool = False,
    ):
        super().__init__()
        self.encoder = model
        self.decoder = Linear(hidden_channels, out_channels)
        self.sheaf = sheaf_model
        self.task = task

        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        elif pooling == "attention":
            self.pool = AttentionPooling(hidden_channels, heads=1)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        metrics_params = {
            "task": task,
            "num_labels": out_channels,
            "num_classes": out_channels,
        }

        self.train_metrics = MetricCollection(
            {
                "micro-f1": F1Score(average="micro", **metrics_params),
                "macro-f1": F1Score(average="macro", **metrics_params),
                "accuracy": Accuracy(**metrics_params),
                "auroc": AUROC(**metrics_params),
            },
            prefix="train/",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        if task == "binary":
            self.loss_fn: Callable = F.binary_cross_entropy_with_logits
            self.act_fn: Callable = torch.sigmoid
        elif task == "multilabel":
            self.loss_fn: Callable = F.multilabel_soft_margin_loss
            self.act_fn: Callable = torch.sigmoid
        else:
            self.loss_fn: Callable = F.cross_entropy
            self.act_fn: Callable = functools.partial(F.softmax, dim=-1)

        self.save_hyperparameters(ignore=["model"])

    def predict_proba(self, data: Data) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            node_embeddings = self.encoder(data)
            # Handle sheaf models that return (embeddings, maps)
            if isinstance(node_embeddings, tuple):
                node_embeddings = node_embeddings[0]
            graph_embeddings = self.pool(node_embeddings, data.batch)
            logits = self.decoder(graph_embeddings)
            if self.task == "binary":
                logits = logits.squeeze(-1)
            probs = self.act_fn(logits)
        return probs

    def common_step(self, data: Data) -> CommonStepOutput:
        # Encode
        node_embeddings = self.encoder(data)
        # Handle sheaf models that return (embeddings, maps)
        if isinstance(node_embeddings, tuple):
            node_embeddings = node_embeddings[0]
        
        # Pool
        graph_embeddings = self.pool(node_embeddings, data.batch) # batch is always present due to dataset transform to ensure global pooling works
        
        # Decode
        y_hat = self.decoder(graph_embeddings)

        # Binary: squeeze before loss
        if self.task == "binary":
            y_hat = y_hat.squeeze(-1)
            loss = self.loss_fn(y_hat, data.y.float())
        elif self.task == "multilabel":
            loss = self.loss_fn(y_hat, data.y.float())
        else:
            loss = self.loss_fn(y_hat, data.y)

        y_hat = self.act_fn(y_hat)
        return CommonStepOutput(y=data.y.int(), y_hat=y_hat, loss=loss)

    def training_step(self, data: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(data)
        self.train_metrics.update(y_hat, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, data: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(data)
        self.valid_metrics.update(y_hat, y)
        self.log("valid/loss", loss, on_epoch=True, batch_size=1)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, data: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss = self.common_step(data)
        self.test_metrics.update(y_hat, y)
        self.log("test/loss", loss, on_epoch=True, batch_size=1)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid/loss"},
        }