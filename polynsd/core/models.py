#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from enum import auto
from typing import Union, Type

from strenum import UppercaseStrEnum

from polynsd.core.sheaf_configs import ModelTypes
from polynsd.datasets.hgb import HGBBaseDataModule
from polynsd.datasets.link_pred import LinkPredBase
from polynsd.models.gnn_baselines import (
    HAN,
    HGT,
    RGCN,
    GCN,
    GAT,
    GIN,
    GraphSAGE,
    HeteroGAT,
    HeteroGraphSAGE,
)
from polynsd.models.sheaf_gnn import (
    DiscreteDiagSheafDiffusion,
    DiscreteBundleSheafDiffusion,
    DiscreteGeneralSheafDiffusion,
    DiscreteSheafDiffusion,
    DiscreteDiagSheafDiffusionPolynomial,
    DiscreteBundleSheafDiffusionPolynomial,
    DiscreteGeneralSheafDiffusionPolynomial,
)
from polynsd.models.sheaf_gnn.inductive import (
    InductiveDiscreteDiagSheafDiffusion,
    InductiveDiscreteBundleSheafDiffusion,
    InductiveDiscreteGeneralSheafDiffusion,
    InductivePolynomialDiscreteDiagSheafDiffusion,
    InductivePolynomialDiscreteBundleSheafDiffusion,
    InductivePolynomialDiscreteGeneralSheafDiffusion,
)


class Models(UppercaseStrEnum):
    HAN = auto()
    HGT = auto()
    RGCN = auto()
    GCN = auto()
    GAT = auto()
    GIN = auto()
    SAGE = auto()
    HETEROGAT = auto()
    HETEROSAGE = auto()


def get_baseline_model(
    model: Models,
    datamodule: Union[HGBBaseDataModule, LinkPredBase],
    hidden_channels: int = 256,
):
    if model == Models.HAN:
        return (
            HAN(
                datamodule.metadata,
                in_channels=datamodule.in_channels,
                hidden_channels=hidden_channels,
            ),
            False,
        )
    elif model == Models.HGT:
        return (
            HGT(
                datamodule.metadata,
                in_channels=datamodule.in_channels,
                hidden_channels=hidden_channels,
            ),
            False,
        )
    elif model == Models.RGCN:
        return (
            RGCN(
                hidden_channels=hidden_channels,
                num_nodes=datamodule.num_nodes,
                num_relations=len(datamodule.metadata[1]),
            ),
            False,
        )
    elif model == Models.GCN:
        gcn = GCN(hidden_channels=hidden_channels, in_channels=datamodule.in_channels)
        return gcn, True
    elif model == Models.GAT:
        gat = GAT(hidden_channels=hidden_channels, in_channels=datamodule.in_channels)
        return gat, True
    elif model == Models.GIN:
        gin = GIN(hidden_channels=hidden_channels, in_channels=datamodule.in_channels)
        return gin, True
    elif model == Models.SAGE:
        sage = GraphSAGE(
            hidden_channels=hidden_channels, in_channels=datamodule.in_channels
        )
        return sage, True
    elif model == Models.HETEROGAT:
        return (
            HeteroGAT(
                datamodule.metadata,
                in_channels=datamodule.in_channels,
                hidden_channels=hidden_channels,
            ),
            False,
        )
    elif model == Models.HETEROSAGE:
        return (
            HeteroGraphSAGE(
                datamodule.metadata,
                in_channels=datamodule.in_channels,
                hidden_channels=hidden_channels,
            ),
            False,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def get_sheaf_model(model: ModelTypes) -> Type[DiscreteSheafDiffusion]:
    if model == ModelTypes.DiagSheaf:
        return DiscreteDiagSheafDiffusion
    if model == ModelTypes.BundleSheaf:
        return DiscreteBundleSheafDiffusion
    if model == ModelTypes.GeneralSheaf:
        return DiscreteGeneralSheafDiffusion
    # if model == ModelTypes.DiagSheafODE:
    #     return DiagSheafDiffusion
    # if model == ModelTypes.BundleSheafODE:
    #     return BundleSheafDiffusion
    # if model == ModelTypes.GeneralSheafODE:
    #     return GeneralSheafDiffusion
    if model == ModelTypes.DiagSheafPoly:
        return DiscreteDiagSheafDiffusionPolynomial
    if model == ModelTypes.BundleSheafPoly:
        return DiscreteBundleSheafDiffusionPolynomial
    if model == ModelTypes.GeneralSheafPoly:
        return DiscreteGeneralSheafDiffusionPolynomial

    raise ValueError(f"Unknown model type: {model}")


def get_inductive_sheaf_model(model: ModelTypes) -> Type[DiscreteSheafDiffusion]:
    if model == ModelTypes.DiagSheaf:
        return InductiveDiscreteDiagSheafDiffusion
    if model == ModelTypes.BundleSheaf:
        return InductiveDiscreteBundleSheafDiffusion
    if model == ModelTypes.GeneralSheaf:
        return InductiveDiscreteGeneralSheafDiffusion
    if model == ModelTypes.DiagSheafPoly:
        return InductivePolynomialDiscreteDiagSheafDiffusion
    if model == ModelTypes.BundleSheafPoly:
        return InductivePolynomialDiscreteBundleSheafDiffusion
    if model == ModelTypes.GeneralSheafPoly:
        return InductivePolynomialDiscreteGeneralSheafDiffusion

    raise ValueError(f"Unknown model type: {model}")
