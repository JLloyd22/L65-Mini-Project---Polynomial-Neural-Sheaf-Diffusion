#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from enum import auto

from strenum import UppercaseStrEnum, PascalCaseStrEnum

from polynsd.datasets.hgb import (
    HGBBaseDataModule,
    DBLPDataModule,
    ACMDataModule,
    IMDBDataModule,
)
from polynsd.datasets.hgt import (
    HGTBaseDataModule,
    HGTDBLPDataModule,
    HGTACMDataModule,
    HGTIMDBDataModule,
)
from polynsd.datasets.link_pred import (
    LinkPredBase,
    LastFMDataModule,
    AmazonBooksDataModule,
    MovieLensDataModule,
)
from polynsd.datasets.biomedical_dataset import (
    CTD_DDADataModule,
    NDFRT_DDADataModule,
    DrugBankDDIDataModule,
    STRINGPPIDataModule,
    NeoDTINetDataModule,
    DeepDRNetDataModule,
)


class NCDatasets(UppercaseStrEnum):
    DBLP = auto()
    ACM = auto()
    IMDB = auto()


def get_dataset_nc(dataset: NCDatasets, homogeneous: bool = False) -> HGBBaseDataModule:
    if dataset == NCDatasets.DBLP:
        return DBLPDataModule(homogeneous=homogeneous)
    elif dataset == NCDatasets.ACM:
        return ACMDataModule(homogeneous=homogeneous)
    elif dataset == NCDatasets.IMDB:
        return IMDBDataModule(homogeneous=homogeneous)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_dataset_hgt(dataset: NCDatasets) -> HGTBaseDataModule:
    if dataset == NCDatasets.DBLP:
        return HGTDBLPDataModule()
    elif dataset == NCDatasets.ACM:
        return HGTACMDataModule()
    else:
        return HGTIMDBDataModule()


class LinkPredDatasets(PascalCaseStrEnum):
    LastFM = "LastFM"
    AmazonBooks = auto()
    MovieLens = auto()
    # Biomedical datasets
    CTD_DDA = "CTD_DDA"
    NDFRT_DDA = "NDFRT_DDA"
    DrugBankDDI = "DrugBankDDI"
    STRINGPPI = "STRINGPPI"
    NeoDTINet = "NeoDTINet"
    DeepDRNet = "DeepDRNet"


def get_dataset_lp(
    dataset: LinkPredDatasets, is_homogeneous: bool = False
) -> LinkPredBase:
    if dataset == LinkPredDatasets.LastFM:
        return LastFMDataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.AmazonBooks:
        return AmazonBooksDataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.MovieLens:
        return MovieLensDataModule(homogeneous=is_homogeneous)
    # Biomedical datasets
    elif dataset == LinkPredDatasets.CTD_DDA:
        return CTD_DDADataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.NDFRT_DDA:
        return NDFRT_DDADataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.DrugBankDDI:
        return DrugBankDDIDataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.STRINGPPI:
        return STRINGPPIDataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.NeoDTINet:
        return NeoDTINetDataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.DeepDRNet:
        return DeepDRNetDataModule(homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.PubMed:
        raise ValueError(f"Unknown dataset: {dataset}")
