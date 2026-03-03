from lightning import Trainer, Callback
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

from lightning.pytorch.loggers import WandbLogger
from transformers import Trainer

from polynsd.graph_classification.graph_classifier import GraphClassifier
from polynsd.models.xgnn import Generator
from polynsd.xgnn import XGNNTrainer, visualize_generated_graph
from polynsd.xgnn.trainer import TrainingConfig
from polynsd.xgnn.visualization import visualize_generation_process
from utils.instantiators import instantiate_callbacks, instantiate_loggers, setup_torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="../configs", config_name="xgnn_config")
def main(cfg: DictConfig):
    # Prepare torch and reproducibility settings
    setup_torch()

    # Instantiate datamodule from cfg (uses polynsd.datasets.graph_classification.*DataModule)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    # ensure splits are created if needed
    datamodule.setup()

    dataset = datamodule.dataset
    
    # # Setup loggers (WandB) if configured
    loggers = instantiate_loggers(cfg.get("logger")) if cfg.get("logger", None) else []
    if loggers:
        assert isinstance(loggers[0], WandbLogger)
        # push full config to W&B for reproducibility
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            loggers[0].experiment.config.update(cfg_dict)
        except Exception:
            # Best-effort: set some fields
            loggers[0].experiment.config["dataset"] = f"{datamodule}"
            loggers[0].experiment.config["classifier_save_path"] = cfg.classifier.save_path
            loggers[0].experiment.config["generator_episodes"] = cfg.generator.episodes
    
    # Setup device
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset already prepared via datamodule
    logger.info(f"Loaded datamodule {datamodule}, dataset: {len(dataset)} graphs, {dataset.num_classes} classes, {dataset.num_features} features")
    
    model_kwargs = {
        "in_channels": datamodule.num_node_features,
        "hidden_channels": cfg.classifier.hidden_channels,
        "num_layers": cfg.classifier.get("num_layers", 4),
    }

    # Setup callbacks
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Train classifier
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=loggers, callbacks=callbacks #loggers no logger
    )

    # check if the model is already trained and load it
    if cfg.classifier.get("load_path", None):
        load_path = Path(cfg.classifier.load_path)
        if load_path.is_file():
            logger.info(f"Loading pre-trained classifier model from {load_path}...")
            model = torch.load(load_path, map_location=device)
        else:
            logger.warning(f"Specified load_path {load_path} does not exist or is not a file. Training a new model.")
    else: 
        model = hydra.utils.instantiate(cfg.model, **model_kwargs)

        # Get the model's output dimension (for new GNN baselines)
        model_output_dim = getattr(model, 'output_dim', cfg.classifier.output_dim)
        graph_classifier = GraphClassifier(
            model,
            hidden_channels=model_output_dim,
            out_channels=datamodule.num_classes,
            task=cfg.get("task", "multiclass"),
            pooling=cfg.get("pooling", "mean"),
            sheaf_model=False,
        )

        trainer.fit(graph_classifier, datamodule)
        
    trainer.test(graph_classifier, datamodule)
    
    # Initialize generator
    start_node = cfg.generator.get("start_node", 0) if cfg.generator.get("start_node", 0) >= 0 else None

    generator = Generator(
        classifier      = graph_classifier,
        candidate_set   = cfg.dataset.candidate_set,
        target_class    = cfg.generator.target_class,
        num_classes     = dataset.num_classes,
        max_nodes       = cfg.generator.max_nodes,
        max_gen_steps   = cfg.generator.max_gen_steps,
        num_rollouts    = cfg.generator.num_rollouts,
        hyp_rollout     = cfg.generator.hyp_rollout,
        hyp_rules       = cfg.generator.hyp_rules,
        start_node_idx  = start_node,
        device          = device,
    )

    config = TrainingConfig(
        generator_episodes=cfg.generator.episodes,
        generator_lr=cfg.generator.lr,
        log_interval=cfg.trainer.log_every_n_steps,
    )
    generator_trainer = XGNNTrainer(config=config, device=device)

    
    # Train generator
    logger.info(f"Training generator for class {cfg.generator.target_class}...")
    generator = generator_trainer.train_generator(generator)
    
    # Log basic metrics to WandB if available
    if loggers:
        try:
            loggers[0].experiment.log({"status": "generator_trained", "target_class": args.target_class})
        except Exception:
            pass
    
    # Generate and visualize explanatory graphs
    logger.info(f"Generating {cfg.generation.num_generations} explanatory graphs...")

    for i in range(cfg.generation.num_generations):
        # Generate graph
        generator.eval()
        graph = generator.generate()
        
        # Get predictions
        graph_classifier.eval()
        with torch.no_grad():
            x = graph.feat[:, :-1].to(device)
            # Create Data object for classifier
            num_edges = graph.edge_index.size(1) if graph.edge_index.numel() > 0 else 0
            data = Data(
                x=x, 
                edge_index=graph.edge_index.to(device), 
                batch=torch.zeros(graph.num_nodes, dtype=torch.long, device=device),
                edge_type=torch.zeros(num_edges, dtype=torch.long, device=device),
                node_type=torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            )
            probs = graph_classifier.predict_proba(data)

        logger.info(f"Graph {i + 1}: {graph.num_nodes} nodes, "
                    f"predictions: {probs[0].cpu().numpy()}")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(8, 6))
        visualize_generated_graph(
            generator, graph,
            candidate_set=cfg.dataset.candidate_set,
            ax=ax, show=False,
        )

        if cfg.output.save_plots:
            plot_path = output_dir / f"generated_graph_class{cfg.generator.target_class}_{i + 1}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {plot_path}")
            plt.close(fig)
        else:
            plt.show()
    
    # Visualize generation process for last graph
    logger.info("Visualizing generation process...")
    fig = visualize_generation_process(
        generator,
        candidate_set=cfg.dataset.candidate_set,
    )
    
    if cfg.output.save_plots:
        plot_path = output_dir / f"generation_process_class{cfg.generator.target_class}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {plot_path}")
        plt.close(fig)
    else:
        plt.show()
    
    logger.info("XGNN experiment complete!")


if __name__ == "__main__":
    main()
