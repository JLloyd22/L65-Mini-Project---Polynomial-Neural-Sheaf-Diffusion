#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass
from typing import Union, List

import hydra
from hydra.core.config_store import ConfigStore
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

from polynsd.core.datasets import NCDatasets
from polynsd.core import Models
from polynsd.core import TrainerArgs
from polynsd.datasets.hgb import HGBBaseDataModule
from polynsd.datasets.hgt import HGTBaseDataModule
from polynsd.node_classification import NodeClassifier
from utils.instantiators import instantiate_loggers, instantiate_callbacks, setup_torch


@dataclass
class ModelConfig:
    type: Models = Models.GCN


@dataclass
class DatasetConfig:
    name: NCDatasets = NCDatasets.DBLP


@dataclass
class Config:
    tags: list[str]
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerArgs


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="../configs", config_name="nc_config")
def main(cfg: DictConfig):
    setup_torch()
    datamodule: Union[HGTBaseDataModule, HGBBaseDataModule] = hydra.utils.instantiate(
        cfg.dataset
    )

    datamodule.prepare_data()

    model = hydra.utils.instantiate(
        cfg.model,
        in_channels=datamodule.in_channels,
        num_nodes=datamodule.num_nodes,
        num_relations=len(datamodule.metadata[1]),
        metadata=datamodule.metadata,
    )

    optim_cfg = cfg.get("optimization")
    learning_rate = (
        optim_cfg.get("learning_rate", 1e-3) if optim_cfg is not None else 1e-3
    )
    weight_decay = (
        optim_cfg.get("weight_decay", 0.0) if optim_cfg is not None else 0.0
    )

    # Use model's output_dim if available, otherwise fallback to hidden_channels or 256
    model_output_dim = getattr(model, 'output_dim', getattr(model, 'hidden_channels', 256))

    classifier = NodeClassifier(
        model,
        hidden_channels=model_output_dim,
        target=datamodule.target,
        out_channels=datamodule.num_classes,
        task=datamodule.task,
        homogeneous_model=cfg.dataset.homogeneous,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    print(f"{model}-{datamodule}")

    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    if logger:
        assert isinstance(logger[0], WandbLogger)
        logger[0].experiment.config["model"] = f"{model}"
        logger[0].experiment.config["dataset"] = f"{datamodule}"

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # 5) train the model
    trainer.fit(classifier, datamodule)

    # 6) test the model
    trainer.test(classifier, datamodule)

    timer = next(filter(lambda x: isinstance(x, Timer), callbacks))

    runtime = {
        "train/runtime": timer.time_elapsed("train"),
        "valid/runtime": timer.time_elapsed("validate"),
        "test/runtime": timer.time_elapsed("test"),
    }

    if logger:
        trainer.logger.log_metrics(runtime)
    else:
        print(runtime)


if __name__ == "__main__":
    main()
