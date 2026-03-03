
from __future__ import annotations

from typing import Literal, NamedTuple, TypedDict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_add, scatter_softmax
from lightning.pytorch.utilities.types import STEP_OUTPUT
from polynsd.utils.linalg import batched_inv_sqrt_spd, batched_procrustes_align
from polynsd.models.sheaf_gnn import sheaf_attention as san
from .graph_classifier import GraphClassifier
from polynsd.models.sheaf_gnn.transductive.disc_models import DiscreteSheafDiffusion


class SheafGCStepOutput(NamedTuple):
    y: torch.Tensor
    y_hat: torch.Tensor
    loss: torch.Tensor
    maps: torch.Tensor


class TrainStepOutput(TypedDict):
    loss: torch.Tensor
    restriction_maps: torch.Tensor


# ==================================================
# "Universal" Stalk-Space Late-Token Readout
# ==================================================
class LateTokenSheafReadoutTransport(nn.Module):
    """
    Universal late-token readout that preserves stalk structure through pooling for all map families:
    diagonal / bundle / general.

    Steps (per node i):
      1) Encoder Embedding in input: H_i in R^{d x C}
      2) “Whitening” removes Scaling/Shearing: S_i = H_i H_i^T + eps_whiten I
      3) Procrustes alignment fixes the remaining rotation ambiguity: Z_i = S_i^{-1/2} H_i      
      4) "Alignment" output: Z_i_aligned = ProcrustesAlign(Z_i, anchor)  # fix remaining orth ambiguity into shared frame

    Then, per graph:
      1) We extract attention weights using an invariant descriptor u_i = sum_d Z_i^2  (diag(Z_i^T Z_i))
      2) We pool in the Stalk Space: pooled = sum_i weights_i * Z_i_aligned     in R^{d x C}
         The token update here (receive-only) occurs in stalk space
      3) We output graph features as invariant diag Gram of token: sum_d token^2 -> R^C
    """

    def __init__(
        self,
        stalk_dim: int,
        channel_dim: int,
        hidden_dim: Optional[int] = None,
        eps_whiten: float = 1e-4,
        eps_eigh: float = 1e-6,
        proper_rot: bool = True,
    ):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.channel_dim = channel_dim

        self.eps_whiten = eps_whiten
        self.eps_eigh = eps_eigh
        self.proper_rot = proper_rot

        hidden = hidden_dim or channel_dim

        # Shared anchor (a.k.a. shared stalk frame, or global reference frame) that defines the pooled stalk-frame (learned).
        self.anchor = nn.Parameter(torch.zeros(stalk_dim, channel_dim))
        nn.init.normal_(self.anchor, std=0.02)

        # Attention scoring network on invariant per-node descriptor u in R^C.
        self.score = nn.Sequential(
            nn.Linear(channel_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Learnable token in stalk space.
        self.token = nn.Parameter(torch.zeros(1, stalk_dim, channel_dim))
        nn.init.normal_(self.token, std=0.02)

        # Receive-only token update in stalk space. 
        self.collect = nn.Sequential(
            nn.Linear(channel_dim * 2, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, channel_dim),
        )
        self.layer_norm = nn.LayerNorm(channel_dim)

    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            batch = node_embeddings.new_zeros(node_embeddings.size(0), dtype=torch.long)

        # Accept either [N, d*C] or [N, d, C]
        if node_embeddings.dim() == 2:
            N, feat_dim = node_embeddings.shape
            expected = self.stalk_dim * self.channel_dim
            if feat_dim != expected:
                raise ValueError(
                    f"LateTokenSheafReadoutTransport expected feat_dim={expected}, got {feat_dim}"
                )
            H = node_embeddings.reshape(N, self.stalk_dim, self.channel_dim)  # [N,d,C]
        elif node_embeddings.dim() == 3:
            N, d, C = node_embeddings.shape
            if (d, C) != (self.stalk_dim, self.channel_dim):
                raise ValueError(
                    f"LateTokenSheafReadoutTransport expected [N,{self.stalk_dim},{self.channel_dim}], "
                    f"got {tuple(node_embeddings.shape)}"
                )
            H = node_embeddings
        else:
            raise ValueError(f"Unexpected node_embeddings shape: {tuple(node_embeddings.shape)}")

        # (A) Whitening: Z = (H H^T + eps I)^(-1/2) H
        I = torch.eye(self.stalk_dim, device=H.device, dtype=H.dtype).unsqueeze(0)  # [1,d,d]
        S = H @ H.transpose(1, 2) + self.eps_whiten * I                              # [N,d,d]
        S_inv_sqrt = batched_inv_sqrt_spd(S, eps=self.eps_eigh)                     # [N,d,d]
        Z = S_inv_sqrt @ H                                                          # [N,d,C]

        # (B) Align to common frame via Procrustes. 
        Z_aligned = batched_procrustes_align(Z, self.anchor, proper=self.proper_rot)  # [N,d,C]

        # (C) Invariant descriptor for attention weights: u = diag(Z^T Z) = sum_d Z^2
        u = (Z ** 2).sum(dim=1)                                                     # [N,C]
        scores = self.score(u).squeeze(-1)                                          # [N]
        weights = scatter_softmax(scores, batch)                                    # [N]

        #TODO: we are only doing weighted sum pooling here, we could also try other variants.
        # (D) Pool in stalk space.
        pooled = scatter_add(Z_aligned * weights[:, None, None], batch, dim=0)      # [B,d,C]

        # (E) Receive-only token update in stalk space.
        B = pooled.size(0)
        token = self.token.expand(B, -1, -1)                                        # [B,d,C]
        combined = torch.cat([token, pooled], dim=-1)                               # [B,d,2C]
        updated = self.layer_norm(self.collect(combined))                           # [B,d,C]

        # (F) Output invariant graph features: diag Gram (energy) in R^C.
        graph_features = (updated ** 2).sum(dim=1)                                  # [B,C]
        return graph_features


# ==================================================
# Classifier that ONLY uses the transport late-token readout
# ==================================================
class SheafGraphClassifier(GraphClassifier):
    def __init__(
        self,
        model: DiscreteSheafDiffusion,
        out_channels: int = 2,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        pooling: Optional[Literal["mean", "max", "add", "attention", "late_token"]] = None,
        transport_eps_whiten: float = 1e-4,
        transport_eps_eigh: float = 1e-6,
        transport_proper_rot: bool = True,
        transport_hidden_dim: Optional[int] = None,
    ):
        # Dimensions: assume encoder outputs flattened [N, d*C] where C=model.hidden_channels.
        hidden_dim = model.hidden_dim
        stalk_dim = getattr(model, "final_d", getattr(model, "d", 1))
        channel_dim = getattr(model, "hidden_channels", hidden_dim // max(stalk_dim, 1))

        # Decoder input is always [B, C] from our readout
        requested_pool = pooling or "mean"
        super().__init__(
            model=model,
            hidden_channels=channel_dim,
            out_channels=out_channels,
            task=task,
            pooling=requested_pool,          
            sheaf_model=True,
        )

        self.hidden_dim = hidden_dim
        self.requested_pooling = pooling
        self.stalk_dim = stalk_dim
        self.channel_dim = channel_dim

        self.readout = LateTokenSheafReadoutTransport(
            stalk_dim=stalk_dim,
            channel_dim=channel_dim,
            hidden_dim=transport_hidden_dim,
            eps_whiten=transport_eps_whiten,
            eps_eigh=transport_eps_eigh,
            proper_rot=transport_proper_rot,
        )

        self.save_hyperparameters(ignore=["model"])

    @staticmethod
    def _num_graphs_from_batch(batch: torch.Tensor) -> int:
        if batch.numel() == 0:
            return 0
        return int(batch.max().item()) + 1

    def common_step(self, data: Data) -> SheafGCStepOutput:
        
        # Ensure batch vector exists.
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        # Ensure edge_type and node_type exist (NSD expectation).
        if not hasattr(data, "edge_type") or data.edge_type is None:
            data._store.__dict__["edge_type"] = None
        if not hasattr(data, "node_type") or data.node_type is None:
            data._store.__dict__["node_type"] = None

        # Update encoder for this graph's topology (inductive).
        num_nodes = data.x.size(0)
        if hasattr(self.encoder, "graph_size"):
            self.encoder.graph_size = num_nodes
        if hasattr(self.encoder, "edge_index"):
            self.encoder.edge_index = data.edge_index

        # Update builders if present.
        self.encoder.regenerate_builder(num_nodes, data.edge_index)

        # Forward encoder.
        node_embeddings, maps = self.encoder(data)  # typically [N, d*C]

        #TODO: for now we do energy readout only, we can also try other things like attention or mean over the stalks, or anything in between.
        # Universal stalk-space late-token readout.
        graph_embedding = self.readout(node_embeddings, data.batch)  # [B, C]

        # Decode
        y_hat = self.decoder(graph_embedding)

        # Loss
        if self.task == "binary":
            y_hat = y_hat.squeeze(-1)
            loss = self.loss_fn(y_hat, data.y.float())
        elif self.task == "multilabel":
            loss = self.loss_fn(y_hat, data.y.float())
        else:
            loss = self.loss_fn(y_hat, data.y)

        y_hat = self.act_fn(y_hat)
        return SheafGCStepOutput(y=data.y.int(), y_hat=y_hat, loss=loss, maps=maps)

    def training_step(self, data: Data, batch_idx: int) -> TrainStepOutput:
        y, y_hat, loss, maps = self.common_step(data)
        self.train_metrics.update(y_hat, y)

        bs = getattr(data, "num_graphs", self._num_graphs_from_batch(data.batch))
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return TrainStepOutput(loss=loss, restriction_maps=maps)

    def predict_proba(self, data: Data) -> torch.Tensor:
        """Override predict_proba to use sheaf-specific readout."""
        self.eval()
        with torch.no_grad():
            # Ensure batch vector exists
            if not hasattr(data, "batch") or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            
            # Ensure edge_type and node_type exist (default to 0 for homogeneous graphs)
            num_edges = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0
            if not hasattr(data, "edge_type") or data.edge_type is None:
                data.edge_type = torch.zeros(num_edges, dtype=torch.long, device=data.x.device)
            if not hasattr(data, "node_type") or data.node_type is None:
                data.node_type = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            
            # Update encoder for this graph's topology
            num_nodes = data.x.size(0)
            if hasattr(self.encoder, "graph_size"):
                self.encoder.graph_size = num_nodes
            if hasattr(self.encoder, "edge_index"):
                self.encoder.edge_index = data.edge_index
            
            # Update builders if present
            self.encoder.regenerate_builder(num_nodes, data.edge_index)
            
            # Forward encoder
            node_embeddings, _ = self.encoder(data)
            
            # Universal stalk-space late-token readout
            graph_embedding = self.readout(node_embeddings, data.batch)
            
            # Decode
            logits = self.decoder(graph_embedding)
            if self.task == "binary":
                logits = logits.squeeze(-1)
            probs = self.act_fn(logits)
        return probs

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, data: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, _ = self.common_step(data)
        self.valid_metrics.update(y_hat, y)
        bs = getattr(data, "num_graphs", self._num_graphs_from_batch(data.batch))
        self.log("valid/loss", loss, on_epoch=True, batch_size=bs)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, data: Data, batch_idx: int) -> STEP_OUTPUT:
        y, y_hat, loss, _ = self.common_step(data)
        self.test_metrics.update(y_hat, y)
        bs = getattr(data, "num_graphs", self._num_graphs_from_batch(data.batch))
        self.log("test/loss", loss, on_epoch=True, batch_size=bs)
        return loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
