#  Copyright (c) 2024 
#  Adapted from X-GNN: Model-Explanations of GNNs using RL
#  License: CC0 1.0 Universal (CC0 1.0)
"""
XGNN Training utilities.

This module provides training functionality for the XGNN generator and classifier.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

if TYPE_CHECKING:
    from torch_geometric.data import Dataset
    from polynsd.models.xgnn import Generator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for XGNN training."""
    
    # Classifier training
    classifier_lr: float = 0.001
    classifier_epochs: int = 1000
    classifier_batch_size: int = 32
    classifier_weight_decay: float = 5e-4
    
    # Generator training
    generator_lr: float = 0.003
    generator_episodes: int = 100
    generator_betas: tuple[float, float] = (0.9, 0.999)
    
    # Logging
    log_interval: int = 10
    
    # Paths
    classifier_save_path: str = "./gcn_classifier.pth"


class XGNNTrainer:
    """
    Trainer for XGNN components.
    
    Handles training of both the GCN classifier and the generator.
    """
    
    def __init__(
        self,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # print config
        logger.info(f"Training configuration: {self.config}")

    def train_classifier(
        self,
        classifier: nn.Module,
        dataset: "Dataset",
        save_path: str | Path | None = None,
    ) -> nn.Module:
        """
        Train the GCN classifier on a graph classification dataset.
        
        Args:
            classifier: GCNClassifier model
            dataset: PyG dataset for training
            save_path: Path to save the trained model
            
        Returns:
            Trained classifier
        """
        save_path = save_path or self.config.classifier_save_path
        
        classifier = classifier.to(self.device)
        classifier.train()
        
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.classifier_batch_size,
            shuffle=True,
        )
        
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=self.config.classifier_lr,
            weight_decay=self.config.classifier_weight_decay,
        )
        
        logger.info(f"Training classifier for {self.config.classifier_epochs} epochs...")
        
        for epoch in range(self.config.classifier_epochs):
            total_loss = 0.0
            
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                out = classifier(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(out, data.y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * data.num_graphs
            
            avg_loss = total_loss / len(dataset)
            
            if epoch % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config.classifier_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on training set
        classifier.eval()
        correct = 0
        
        with torch.no_grad():
            for data in train_loader:
                data = data.to(self.device)
                out = classifier(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
        
        accuracy = correct / len(dataset)
        logger.info(f"Training Set Accuracy: {accuracy:.4f}")
        
        # Save model
        torch.save(classifier.state_dict(), save_path)
        logger.info(f"Classifier saved to {save_path}")
        
        return classifier
    
    def train_generator(
        self,
        generator: "Generator",
    ) -> "Generator":
        """
        Train the generator using reinforcement learning.
        
        Args:
            generator: Generator model (with classifier already set)
            
        Returns:
            Trained generator
        """
        generator = generator.to(self.device)
        generator.train()
        
        optimizer = optim.Adam(
            generator.parameters(),
            lr=self.config.generator_lr,
            betas=self.config.generator_betas,
        )
        
        logger.info(f"Training generator for {self.config.generator_episodes} episodes...")
        
        # Tracking metrics
        reward_sum = 0.0
        loss_sum = 0.0
        step_count = 0
        
        for episode in range(self.config.generator_episodes):
            # Reset graph
            generator.reset_graph()
            
            for step in range(generator.max_gen_steps):
                # Store current graph state
                current_graph = generator.current_graph.clone()
                
                # Generate next step
                output = generator.forward(current_graph)
                
                # Calculate reward
                reward, is_valid = generator.calculate_reward(output.graph)
                reward_sum += reward.item()
                step_count += 1
                
                # Calculate and apply loss
                loss = generator.calculate_loss(reward, output)
                loss_sum += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update graph if valid
                # if reward > 0: 
                if is_valid:
                    generator.current_graph = output.graph
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = reward_sum / max(step_count, 1)
                avg_loss = loss_sum / max(step_count, 1)
                logger.info(
                    f"Episode {episode + 1}/{self.config.generator_episodes}, "
                    f"Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}"
                )
                reward_sum = 0.0
                loss_sum = 0.0
                step_count = 0
        
        logger.info("Generator training complete.")
        return generator
    
    def load_classifier(
        self,
        classifier: nn.Module,
        path: str | Path,
    ) -> nn.Module:
        """Load a pre-trained classifier."""
        classifier.load_state_dict(torch.load(path, map_location=self.device))
        classifier = classifier.to(self.device)
        logger.info(f"Classifier loaded from {path}")
        return classifier
