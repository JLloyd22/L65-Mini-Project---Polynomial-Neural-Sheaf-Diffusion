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

from polynsd.graph_classification import SheafGraphClassifier
from polynsd.models.xgnn import SheafGeneratorPROTEINS, SheafGeneratorMUTAG
from polynsd.xgnn import XGNNTrainer, visualize_sheaf_generated_graph
from polynsd.xgnn.trainer import TrainingConfig
from polynsd.xgnn.visualization import visualize_sheaf_generation_process

from utils.instantiators import instantiate_callbacks, instantiate_loggers, setup_torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="../configs", config_name="sheaf_xgnn_config")
def main(cfg: DictConfig):
    # Prepare torch and reproducibility settings
    setup_torch()

    # Convert relative paths to absolute (Hydra changes CWD)
    cfg_output_dir = hydra.utils.to_absolute_path(cfg.output.output_dir)

    # Instantiate datamodule from cfg (uses polynsd.datasets.graph_classification.*DataModule)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    # ensure splits are created if needed
    datamodule.setup("fit")
    dataset_name = cfg.dataset._target_.lower().replace("datamodule", "").split(".")[-1]
    logger.info(f"dataset {dataset_name} prepared with task: {cfg.dataset['task']}")

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO: what dis?
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset already prepared via datamodule
    logger.info(f"Loaded datamodule {datamodule}, \n dataset: {len(dataset)} graphs, {dataset.num_classes} classes, {dataset.num_features} features")
    
    # Calculate graph_size as the max number of nodes in a single graph
    max_nodes = 0
    for graph in datamodule.train_dataset:
        max_nodes = max(max_nodes, graph.num_nodes)
    graph_size = max_nodes    
    
    # Merge model kwargs into config.model.args
    from omegaconf import OmegaConf, open_dict
    with open_dict(cfg.model):
        if hasattr(cfg.model, 'args'):
            with open_dict(cfg.model.args):
                cfg.model.args.input_dim = datamodule.num_node_features
                cfg.model.args.hidden_channels = cfg.classifier.hidden_channels
                cfg.model.args.output_dim = cfg.classifier.output_dim
                cfg.model.args.graph_size = graph_size
                cfg.model.args.num_edge_types = getattr(datamodule, "num_edge_types", 1)
                cfg.model.args.num_node_types = getattr(datamodule, "num_node_types", 1)

    # create unique filename (extract model class name from _target_)
    model_class_name = cfg.model._target_.split(".")[-1] if "_target_" in cfg.model else "model"
    sheaf_learner_name = cfg.sheaf_learner if isinstance(cfg.sheaf_learner, str) else "default"
    classifier_filename = f"xgnn_classifier_{dataset_name}_{model_class_name}_{sheaf_learner_name}_{cfg.classifier.hidden_channels}h_{cfg.classifier.output_dim}o_{cfg.classifier.num_layers}l_{datamodule.num_classes}oc_{cfg.dataset.get('task', 'multiclass')}_task.pth"

    # Setup callbacks
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Train classifier
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=loggers, callbacks=callbacks #loggers no logger
    )

    graph_classifier = None
    # check if the model is already trained and load it
    model = hydra.utils.instantiate(cfg.model)

    # Get the model's output dimension (for new GNN baselines)
    # model_output_dim = getattr(model, 'output_dim', cfg.classifier.output_dim)
    logger.info(f"Model task type: {cfg.dataset.task}")
    graph_classifier = SheafGraphClassifier(
        model,
        out_channels=datamodule.num_classes,
        task=cfg.dataset.get("task", "multiclass"), #TODO: why multiclass? should be in dataset config?
        pooling=cfg.get("pooling", "mean"),
    )


    if cfg.classifier.get("load_path", None):
        # Load the model state dict
        load_path = Path(cfg.classifier.load_path + classifier_filename)
        logger.info(f"Looking for pre-trained classifier model at {load_path}")

        # Check if the file exists
        if not load_path.is_file():
            logger.info(f"No pre-trained classifier model found at {load_path}, training a new model...")
            trainer.fit(graph_classifier, datamodule)  # Train the model if the file doesn't exist
            torch.save(graph_classifier.state_dict(), load_path.with_suffix(".pth"))  # Save the state dict for future use
            logger.info(f"Saved trained classifier model to {load_path}")
            
        else:
            state_dict = torch.load(load_path, map_location=device)
            graph_classifier.load_state_dict(state_dict)
            graph_classifier.to(device)
            logger.info(f"Loaded pre-trained classifier model from {load_path}")    
        
    else:
        trainer.fit(graph_classifier, datamodule)
        
        # graph classifier is a LightningModule, so we can save it using its built-in method
        if cfg.classifier.get("save_path", None):
            save_path = Path(cfg.classifier.save_path + classifier_filename)
            torch.save(graph_classifier.state_dict(), save_path.with_suffix(".pth")) # save the state dict separately for loading into generator
            logger.info(f"Saved trained classifier model to {save_path}")

    

    # trainer.fit(graph_classifier, datamodule)
    trainer.test(graph_classifier, datamodule)
    
    # Initialize generator
    start_node = cfg.generator.get("start_node", 0) if cfg.generator.get("start_node", 0) >= 0 else None

    if dataset_name == "mutag":
        generator = SheafGeneratorMUTAG(
            classifier=graph_classifier,
            num_classes=dataset.num_classes,
            max_nodes=cfg.generator.max_nodes,
            min_nodes=cfg.generator.min_nodes,
            temperature=cfg.generator.get("temperature", 1.0),
            max_gen_steps=cfg.generator.max_gen_steps,
            num_rollouts=cfg.generator.num_rollouts,
            hyp_rollout=cfg.generator.hyp_rollout,
            hyp_rules=cfg.generator.hyp_rules,
            start_node_idx=start_node,
            device=device,
            model_config=cfg.model,
            sheaf_settings=cfg.generator.get("sheaf", None),
            hidden_dim=cfg.generator.get("hidden_dim", 32),
            num_node_types=cfg.model.args.num_node_types,
            num_edge_types=cfg.model.args.num_edge_types,
        )
    elif dataset_name == "proteins":
        generator = SheafGeneratorPROTEINS(
            classifier=graph_classifier,
            num_classes=dataset.num_classes,
            max_nodes=cfg.generator.max_nodes,
            min_nodes=cfg.generator.min_nodes,
            temperature=cfg.generator.get("temperature", 1.0),
            max_gen_steps=cfg.generator.max_gen_steps,
            num_rollouts=cfg.generator.num_rollouts,
            hyp_rollout=cfg.generator.hyp_rollout,
            hyp_rules=cfg.generator.hyp_rules,
            start_node_idx=start_node,
            device=device,
            model_config=cfg.model,
            sheaf_settings=cfg.generator.get("sheaf", None),
            hidden_dim=cfg.generator.get("hidden_dim", 32),
            num_node_types=cfg.model.args.num_node_types,
            num_edge_types=cfg.model.args.num_edge_types,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name} for sheaf generator")

    logger.info(f"Initialized {generator.__class__.__name__} with sheaf model: {generator.sheaf_model.__class__.__name__}")

    config = TrainingConfig(
        generator_episodes=cfg.generator.episodes,
        generator_lr=cfg.generator.lr,
        log_interval=cfg.trainer.log_every_n_steps,
    )
    generator_trainer = XGNNTrainer(config=config, device=device)

    # Create unique generator filename
    generator_filename = f"xgnn_generator_{dataset_name}_{model_class_name}_{sheaf_learner_name}_{cfg.generator.episodes}ep.pth"

    if cfg.generator.get("load_path", None):
        # Load the generator state dict
        load_path = Path(cfg.generator.load_path + generator_filename)
        logger.info(f"Looking for pre-trained generator model at {load_path}")

        # Check if the file exists
        if not load_path.is_file():
            logger.info(f"No pre-trained generator model found at {load_path}, training a new model...")
            generator = generator_trainer.train_generator(generator)
            torch.save(generator.state_dict(), load_path)
            logger.info(f"Saved trained generator model to {load_path}")
        else:
            state_dict = torch.load(load_path, map_location=device)
            generator.load_state_dict(state_dict)
            generator.to(device)
            logger.info(f"Loaded pre-trained generator model from {load_path}")
    else:
        # Train generator
        logger.info(f"Training generator for class {generator.target_class}...")
        generator = generator_trainer.train_generator(generator)
        
        # Save generator if save_path is configured
        if cfg.generator.get("save_path", None):
            save_path = Path(cfg.generator.save_path + generator_filename)
            torch.save(generator.state_dict(), save_path)
            logger.info(f"Saved trained generator model to {save_path}")
            
    # Generate and visualize explanatory graphs
    logger.info(f"Generating {cfg.generation.num_generations} explanatory graphs...")

    for i in range(cfg.generation.num_generations):
        # Generate graph
        generator.eval()
        graph = generator.generate()
        
        # Get predictions
        graph_classifier.eval()
        prob_list = []
        with torch.no_grad():
            probs = generator.get_probs(graph, graph_classifier)[0].cpu().numpy()
            prob_list.append(probs)


        logger.info(f"Graph {i + 1}: {graph.num_nodes} nodes, "
                    f"predictions: {probs}, ")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(8, 6))
        visualize_sheaf_generated_graph(
            generator, graph,
            ax=ax, show=False, probs=probs
        )

        if cfg.output.save_plots:
            plot_path = output_dir / dataset_name / f"generated_graph_class{generator.target_class}_{i + 1}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {plot_path}")
            plt.close(fig)
        else:
            plt.show()
    
    # Visualize generation process
    for i in range(cfg.generation.num_generations):
        logger.info("Visualizing generation process...")
        fig = visualize_sheaf_generation_process(generator)
        
        if cfg.output.save_plots:
            plot_path = output_dir / dataset_name / f"generation_process_class{generator.target_class}_{i + 1}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {plot_path}")
            plt.close(fig)
        else:
            plt.show()
    
    logger.info("XGNN experiment complete!")


if __name__ == "__main__":
    main()
