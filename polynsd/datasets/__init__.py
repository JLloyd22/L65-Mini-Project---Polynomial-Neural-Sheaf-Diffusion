#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

# Biomedical datasets
from .biomedical_dataset import (
    BiomedicalDataset,
    CTD_DDADataModule,
    NDFRT_DDADataModule,
    DrugBankDDIDataModule,
    STRINGPPIDataModule,
    NeoDTINetDataModule,
    DeepDRNetDataModule,
)

# Link prediction datasets
from .link_pred import (
    LinkPredBase,
    LastFMDataModule,
    AmazonBooksDataModule,
    MovieLensDataModule,
    PubMedHNEDataModule,
    AmazonGATNEDataModule,
)

# HGB node classification datasets
from .hgb import (
    HGBBaseDataModule,
    IMDBDataModule,
    DBLPDataModule,
    ACMDataModule,
    FreebaseDataModule,
)

# HGT node classification datasets
from .hgt import (
    HGTBaseDataModule,
    HGTIMDBDataModule,
    HGTDBLPDataModule,
    HGTACMDataModule,
    HGTFreebaseDataModule,
)

# Homogeneous Graph classification datasets
from .graph_classification import (
    GraphClassificationDataModule,
    MUTAGDataModule,
    PROTEINSDataModule,
    ENZYMESDataModule,
    NCI1DataModule,
    NCI109DataModule,
    PTCMRDataModule,
    COLLABDataModule,
    IMDBBinaryDataModule,
    IMDBMultiDataModule,
    REDDITBinaryDataModule,
    REDDIT5KDataModule,
    DDDataModule,
)

__all__ = [
    # Biomedical datasets
    "BiomedicalDataset",
    "CTD_DDADataModule",
    "NDFRT_DDADataModule",
    "DrugBankDDIDataModule",
    "STRINGPPIDataModule",
    "NeoDTINetDataModule",
    "DeepDRNetDataModule",
    # Link prediction base and datasets
    "LinkPredBase",
    "LastFMDataModule",
    "AmazonBooksDataModule",
    "MovieLensDataModule",
    "PubMedHNEDataModule",
    "AmazonGATNEDataModule",
    # HGB node classification
    "HGBBaseDataModule",
    "IMDBDataModule",
    "DBLPDataModule",
    "ACMDataModule",
    "FreebaseDataModule",
    # HGT node classification
    "HGTBaseDataModule",
    "HGTIMDBDataModule",
    "HGTDBLPDataModule",
    "HGTACMDataModule",
    "HGTFreebaseDataModule",
    # Graph classification
    "GraphClassificationDataModule",
    "MUTAGDataModule",
    "PROTEINSDataModule",
    "ENZYMESDataModule",
    "NCI1DataModule",
    "NCI109DataModule",
    "PTCMRDataModule",
    "COLLABDataModule",
    "IMDBBinaryDataModule",
    "IMDBMultiDataModule",
    "REDDITBinaryDataModule",
    "REDDIT5KDataModule",
    "DDDataModule",
]
