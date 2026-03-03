#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass
from enum import auto

from strenum import PascalCaseStrEnum

from polynsd.core.datasets import NCDatasets, LinkPredDatasets


class ModelTypes(PascalCaseStrEnum):
    DiagSheaf = auto()
    BundleSheaf = auto()
    GeneralSheaf = auto()
    DiagSheafODE = auto()
    BundleSheafODE = auto()
    GeneralSheafODE = auto()
    DiagSheafPoly = auto()
    BundleSheafPoly = auto()
    GeneralSheafPoly = auto()


@dataclass
class SheafModelCfg:
    type: ModelTypes = ModelTypes.BundleSheaf


@dataclass
class SheafNCDatasetCfg:
    name: NCDatasets = NCDatasets.DBLP


@dataclass
class SheafLinkPredDatasetCfg:
    name: LinkPredDatasets = LinkPredDatasets.LastFM
