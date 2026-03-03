#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import List

import hydra
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import WandbLogger, Logger
from omegaconf import DictConfig

from polynsd.graph_classification import SheafGraphClassifier
from utils.instantiators import instantiate_loggers, instantiate_callbacks, setup_torch


@hydra.main(version_base="1.2", config_path="../configs", config_name="attention_sheaf_config_gc")
def main(cfg: DictConfig) -> None:

    setup_torch()
    
    # 1) Get the datamodule
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    # Calculate graph_size as the max number of nodes in a single graph
    max_nodes = 0
    for graph in datamodule.train_dataset:
        max_nodes = max(max_nodes, graph.num_nodes)
    graph_size = max_nodes
    
    # 2) Instantiate the sheaf model (inductive models don't need edge_index in init)
    model = hydra.utils.instantiate(
        cfg.model,
        args={
            "graph_size": graph_size,
            "input_dim": datamodule.num_node_features,
            "output_dim": cfg.hidden_dim,
            "num_edge_types": getattr(datamodule, "num_edge_types", 1),
            "num_node_types": getattr(datamodule, "num_node_types", 1),
        },
    )

    # 3) Create graph classifier
    out_channels = datamodule.num_classes if datamodule.task != "binary" else 1
    sheaf_gc = SheafGraphClassifier(
        model,
        out_channels=out_channels,
        task=datamodule.task,
        pooling=cfg.get("pooling", "mean"),
    )

    # 4) Setup logging
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    if logger:
        assert isinstance(logger[0], WandbLogger)
        logger[0].experiment.config["model"] = f"{model}"
        logger[0].experiment.config["dataset"] = f"{datamodule}"

    # 5) Setup callbacks
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # 6) Create trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    # 7) Train the model
    trainer.fit(sheaf_gc, datamodule)

    # 8) Test the model
    trainer.test(sheaf_gc, datamodule)

    # 9) Log runtime metrics
    timer = next(filter(lambda x: isinstance(x, Timer), callbacks), None)
    if timer:
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
