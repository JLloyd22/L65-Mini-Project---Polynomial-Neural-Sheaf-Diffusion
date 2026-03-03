#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from typing import List

import hydra
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import WandbLogger, Logger
from omegaconf import DictConfig

from polynsd.graph_classification import GraphClassifier
from utils.instantiators import instantiate_loggers, instantiate_callbacks, setup_torch


@hydra.main(version_base="1.2", config_path="../configs", config_name="gc_config")
def main(cfg: DictConfig) -> None:

    setup_torch()
    
    # 1) Get the datamodule
    datamodule = hydra.utils.instantiate(cfg.dataset)
    
    # Now prepare and setup data
    datamodule.prepare_data()
    
    # 2) Instantiate the GNN model
    datamodule.setup("fit")
    sample_batch = next(iter(datamodule.train_dataloader()))
    
    model_kwargs = {
        "in_channels": datamodule.num_node_features,
        "hidden_channels": cfg.hidden_dim,
        "num_layers": cfg.get("num_layers", 4),
    }
    
    model = hydra.utils.instantiate(cfg.model, **model_kwargs)

    # Get the model's output dimension (for new GNN baselines)
    model_output_dim = getattr(model, 'output_dim', cfg.hidden_dim)

    # 3) Create graph classifier
    graph_classifier = GraphClassifier(
        model,
        hidden_channels=model_output_dim,
        out_channels=datamodule.num_classes,
        task=cfg.get("task", "multiclass"),
        pooling=cfg.get("pooling", "mean"),
        sheaf_model=False,
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
    trainer.fit(graph_classifier, datamodule)

    # 8) Test the model
    trainer.test(graph_classifier, datamodule)

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
