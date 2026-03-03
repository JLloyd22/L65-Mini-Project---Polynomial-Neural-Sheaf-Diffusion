#  Copyright (c) 2024 
#  Adapted from X-GNN: Model-Explanations of GNNs using RL
#  License: CC0 1.0 Universal (CC0 1.0)
"""
XGNN Graph Generator Module.

This module implements the graph generator from the XGNN paper that learns to generate
graph patterns that explain GNN predictions using reinforcement learning.

Reference:
    "XGNN: A Model-Level Explanation for Graph Neural Networks"
    Yuan et al., 2020
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv



@dataclass
class GeneratedGraph:
    """Container for generated graph state."""
    
    feat: torch.Tensor  # Node features [num_nodes, num_features]
    edge_index: torch.Tensor  # Edge indices [2, num_edges]
    num_nodes: int
    degrees: torch.Tensor  # Degree of each node
    
    def clone(self) -> "GeneratedGraph":
        """Create a deep copy of the graph."""
        return GeneratedGraph(
            feat=self.feat.clone(),
            edge_index=self.edge_index.clone(),
            num_nodes=self.num_nodes,
            degrees=self.degrees.clone(),
        )
    
    def to(self, device: torch.device) -> "GeneratedGraph":
        """Move graph to device."""
        return GeneratedGraph(
            feat=self.feat.to(device),
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            degrees=self.degrees.to(device),
        )


@dataclass
class GeneratorOutput:
    """Output from a single generation step."""
    
    p_start_logits: torch.Tensor
    a_start_idx: torch.Tensor
    p_end_logits: torch.Tensor | None  # None if no-op selected
    a_end_idx: torch.Tensor | int  # -1 if no-op selected
    graph: GeneratedGraph


@dataclass
class CandidateNode:
    """Represents a candidate node type with its properties."""
    
    name: str
    max_valency: int
    
    @classmethod
    def from_string(cls, s: str) -> "CandidateNode":
        """Parse from string like 'C.4' -> CandidateNode(name='C', max_valency=4)"""
        parts = s.split('.')
        if len(parts) != 2:
            return cls(name=parts[0], max_valency=67)  # Default valency if not specified
        return cls(name=parts[0], max_valency=int(parts[1]))


class Generator(nn.Module):
    """
    XGNN Graph Generator.
    
    Generates graph patterns that maximize the target GNN's confidence for a given class
    using reinforcement learning.
    
    Args:
        classifier: Pre-trained classifier to explain
        candidate_set: List of candidate node types (e.g., ['C.4', 'N.3', 'O.2'])
        target_class: Class index to generate explanations for
        num_classes: Total number of classes
        max_nodes: Maximum number of nodes in generated graphs
        max_gen_steps: Maximum generation steps
        hyp_rollout: Hyperparameter for rollout reward weighting
        hyp_rules: Hyperparameter for rule-based reward weighting
        num_rollouts: Number of rollout simulations for reward estimation
        hidden_dim: Hidden dimension for generator GCN layers
        dropout: Dropout probability
        start_node_idx: Fixed starting node index (None for random)
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        candidate_set: list[str],
        target_class: int = 0,
        num_classes: int = 2,
        max_nodes: int = 5,
        max_gen_steps: int = 7,
        hyp_rollout: float = 1.0,
        hyp_rules: float = 2.0,
        num_rollouts: int = 10,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        start_node_idx: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        self.device = device or torch.device('cpu')
        self.dropout = dropout
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.max_gen_steps = max_gen_steps
        self.target_class = target_class
        
        # Parse candidate set
        self.candidate_set = [CandidateNode.from_string(c) for c in candidate_set]
        self.candidate_set_raw = candidate_set
        self.num_candidates = len(candidate_set)
        
        # Feature dimension: one-hot for each candidate + 1 for no-op node
        self.num_features = self.num_candidates + 1
        
        # Hyperparameters
        self.hyp_rollout = hyp_rollout
        self.hyp_rules = hyp_rules
        self.num_rollouts = num_rollouts
        
        # Starting node configuration
        self.start_node_idx = start_node_idx
        self.random_start = start_node_idx is None
        
        # Build generator network
        # Input projection
        self.fc_in = nn.Linear(self.num_features, 8)
        
        # GCN layers for node embedding
        self.gc1 = GCNConv(8, 16, add_self_loops=True, normalize=True)
        self.gc2 = GCNConv(16, 24, add_self_loops=True, normalize=True)
        self.gc3 = GCNConv(24, hidden_dim, add_self_loops=True, normalize=True)
        
        # MLP for selecting start node
        self.mlp_start = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # MLP for selecting end node (takes concatenated start and candidate embeddings)
        self.mlp_end = nn.Sequential(
            nn.Linear(hidden_dim * 2, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        
        # Store the classifier (frozen)
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        # Initialize tracking
        self.gen_steps_done = 0
        self.current_graph: GeneratedGraph | None = None
        
        # Move to device
        self.to(self.device)
    
    def reset_graph(self) -> GeneratedGraph:
        """Reset to initial graph with single starting node."""
        self.gen_steps_done = 0
        
        # Select starting node
        if self.random_start:
            start_idx = random.randint(0, self.num_candidates - 1)
        else:
            start_idx = self.start_node_idx
        
        # Create one-hot feature for starting node
        feat = torch.zeros((1, self.num_features), dtype=torch.float32, device=self.device)
        feat[0, start_idx] = 1.0
        
        # Empty edge index
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Initial degree
        degrees = torch.zeros(1, dtype=torch.long, device=self.device)
        
        self.current_graph = GeneratedGraph(
            feat=feat,
            edge_index=edge_index,
            num_nodes=1,
            degrees=degrees,
        )
        
        return self.current_graph
    
    def _compute_node_embeddings(
        self,
        graph: GeneratedGraph,
    ) -> torch.Tensor:
        """
        Compute node embeddings for graph nodes and candidate nodes.
        
        Returns embeddings for: [graph_nodes, candidate_nodes, no_op_node]
        """
        # Create candidate node features (identity matrix + no-op)
        candidate_feats = torch.eye(self.num_features, dtype=torch.float32, device=self.device)
        
        # Concatenate graph features with candidate features
        x = torch.cat([graph.feat, candidate_feats], dim=0)
        
        # Use graph edges (candidates are isolated initially)
        edge_index = graph.edge_index
        
        # Forward through GCN layers
        x = F.relu6(self.fc_in(x))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu6(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu6(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu6(self.gc3(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        
        return x
    
    def forward(self, graph: GeneratedGraph) -> GeneratorOutput:
        """
        Generate next graph state from current graph.
        
        The generation process:
        1. Compute node embeddings for graph + candidate nodes
        2. Select start node from existing graph nodes
        3. Select end node from all nodes (existing + candidates)
        4. Add edge (and potentially new node) to graph
        
        Args:
            graph: Current graph state
            
        Returns:
            GeneratorOutput with probabilities, actions, and new graph
        """
        # Work with a copy
        G = graph.clone()
        
        # Compute embeddings
        x = self._compute_node_embeddings(G)
        
        num_total = G.num_nodes + self.num_candidates + 1  # +1 for no-op
        
        # Create mask for start node selection
        # Only existing graph nodes can be start nodes (not candidates, not no-op)
        # But we allow no-op to be selected (last position)
        mask_start = torch.zeros(num_total, dtype=torch.bool, device=self.device)
        mask_start[G.num_nodes:num_total - 1] = True  # Mask candidates
        mask_start = mask_start.view(-1, 1)
        
        # Compute start node probabilities
        p_start_logits = self.mlp_start(x)
        p_start_logits = p_start_logits - p_start_logits.max()  # Numerical stability
        p_start = F.softmax(p_start_logits.masked_fill(mask_start, -1e8), dim=0)
        p_start = p_start / p_start.sum()  # Renormalize
        
        # Sample start node
        a_start_idx = torch.multinomial(p_start.view(-1), num_samples=1).squeeze()
        
        # Check if no-op selected
        if a_start_idx == num_total - 1:
            return GeneratorOutput(
                p_start_logits=p_start_logits,
                a_start_idx=a_start_idx,
                p_end_logits=None,
                a_end_idx=-1,
                graph=G,
            )
        
        # Compute end node selection
        # Concatenate start node embedding with all node embeddings
        x_start = x[a_start_idx].unsqueeze(0).expand(num_total, -1)
        x_combined = torch.cat([x, x_start], dim=1)
        
        # Create mask for end node selection
        # Cannot select: start node itself, no-op node
        mask_end = torch.zeros(num_total, dtype=torch.bool, device=self.device)
        mask_end[a_start_idx] = True  # Can't connect to self
        mask_end[-1] = True  # Can't select no-op as end
        mask_end = mask_end.view(-1, 1)
        
        # Compute end node probabilities
        p_end_logits = self.mlp_end(x_combined)
        p_end_logits = p_end_logits - p_end_logits.max()  # Numerical stability
        p_end = F.softmax(p_end_logits.masked_fill(mask_end, -1e8), dim=0)
        p_end = p_end / p_end.sum()  # Renormalize
        
        # Sample end node
        a_end_idx = torch.multinomial(p_end.view(-1), num_samples=1).squeeze()
        
        # Update graph
        G = self._add_edge(G, a_start_idx.item(), a_end_idx.item())
        
        return GeneratorOutput(
            p_start_logits=p_start_logits,
            a_start_idx=a_start_idx,
            p_end_logits=p_end_logits,
            a_end_idx=a_end_idx,
            graph=G,
        )
    
    def _add_edge(
        self,
        graph: GeneratedGraph,
        start_idx: int,
        end_idx: int,
    ) -> GeneratedGraph:
        """Add edge to graph, potentially adding a new node."""
        G = graph.clone()
        
        if end_idx >= G.num_nodes:
            # End node is from candidate set - add new node
            # Get the candidate index (accounting for existing nodes)
            candidate_idx = end_idx - G.num_nodes
            
            # Create new node feature
            new_feat = torch.zeros((1, self.num_features), dtype=torch.float32, device=self.device)
            new_feat[0, candidate_idx] = 1.0
            
            # Add to feature matrix
            G.feat = torch.cat([G.feat, new_feat], dim=0)
            
            # Add edges (bidirectional)
            new_edges = torch.tensor(
                [[start_idx, G.num_nodes], [G.num_nodes, start_idx]],
                dtype=torch.long,
                device=self.device,
            )
            G.edge_index = torch.cat([G.edge_index, new_edges], dim=1)
            
            # Update degrees
            G.degrees = torch.cat([G.degrees, torch.ones(1, dtype=torch.long, device=self.device)])
            G.degrees[start_idx] += 1
            
            # Update node count
            G.num_nodes += 1
        else:
            # End node exists - just add edge
            new_edges = torch.tensor(
                [[start_idx, end_idx], [end_idx, start_idx]],
                dtype=torch.long,
                device=self.device,
            )
            G.edge_index = torch.cat([G.edge_index, new_edges], dim=1)
            
            # Update degrees
            G.degrees[start_idx] += 1
            G.degrees[end_idx] += 1
        
        return G
    
    def calculate_reward(
        self,
        graph: GeneratedGraph,
        use_rollout: bool = True,
    ) -> tuple[torch.Tensor, bool]:
        """
        Calculate reward for generated graph.
        
        Reward = R_feedback + hyp_rules * R_rules
        
        Where R_feedback includes:
        - Immediate classifier feedback
        - Averaged rollout simulations
        
        Args:
            graph: Generated graph state
            use_rollout: Whether to use rollout simulations
            
        Returns:
            Total reward (scalar tensor)
        """
        self.gen_steps_done += 1
        
        # Check graph rules
        r_rules = self._check_graph_rules(graph)
        
        # Calculate classifier feedback
        r_feedback = self._calculate_classifier_reward(graph)
        
        # Rollout simulations
        if use_rollout and self.num_rollouts > 0:
            rollout_rewards = []
            
            # Temporarily disable training mode for rollouts
            was_training = self.training
            self.eval()
            
            with torch.no_grad():
                for _ in range(self.num_rollouts):
                    G_roll = graph.clone()
                    
                    # Continue generation until max steps
                    for _ in range(self.max_gen_steps - self.gen_steps_done):
                        output = self.forward(G_roll)
                        G_roll = output.graph
                    
                    rollout_rewards.append(self._calculate_classifier_reward(G_roll))
            
            if was_training:
                self.train()
            
            avg_rollout = sum(rollout_rewards) / len(rollout_rewards)
            r_feedback = r_feedback + self.hyp_rollout * avg_rollout
        
        return r_feedback + self.hyp_rules * r_rules, r_rules >= 0  
    
    def _calculate_classifier_reward(self, graph: GeneratedGraph) -> torch.Tensor:
        """
        Calculate reward from classifier predictions.
        
        Reward = P(target_class | graph) - 1/num_classes
        
        This encourages graphs that increase confidence for target class
        above random chance.
        """
        self.classifier.eval()
        
        with torch.no_grad():
            # Remove the last feature column (no-op indicator) for classifier
            x = graph.feat[:, :-1]
            # Create a Data object for the classifier
            # Add edge_type and node_type for sheaf models (default to 0 for homogeneous graphs)
            num_edges = graph.edge_index.size(1) if graph.edge_index.numel() > 0 else 0
            data = Data(
                x=x, 
                edge_index=graph.edge_index, 
                batch=torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device),
                edge_type=torch.zeros(num_edges, dtype=torch.long, device=x.device),
                node_type=torch.zeros(graph.num_nodes, dtype=torch.long, device=x.device)
            )
            probs = self.classifier.predict_proba(data)
        
        return probs[0, self.target_class] - (1.0 / self.num_classes)
    
    def _check_graph_rules(self, graph: GeneratedGraph) -> torch.Tensor:
        """
        Check if graph satisfies domain rules.
        
        Rules:
        1. No duplicate edges
        2. Node degrees don't exceed valency
        3. No premature no-op (when under max nodes)
        4. Don't exceed max nodes
        
        Returns:
            0 if valid, -1 if invalid
        """
        # Check valency constraints
        for idx, degree in enumerate(graph.degrees):
            if degree > 0:
                node_type_idx = graph.feat[idx].argmax().item()
                if node_type_idx < len(self.candidate_set):
                    max_valency = self.candidate_set[node_type_idx].max_valency
                    if degree > max_valency:
                        return torch.tensor(-1.0, device=self.device)
        
        # Check for duplicate edges
        edges = graph.edge_index.t().tolist()
        edge_set = {tuple(e) for e in edges}
        if len(edge_set) < len(edges):
            return torch.tensor(-1.0, device=self.device)
        
        # Check for premature no-op
        if self.current_graph is not None:
            if (graph.num_nodes == self.current_graph.num_nodes and
                graph.num_nodes < self.max_nodes and
                graph.edge_index.shape[1] == self.current_graph.edge_index.shape[1]):
                return torch.tensor(-1.0, device=self.device)
        
        # Check max nodes
        if graph.num_nodes > self.max_nodes:
            return torch.tensor(-1.0, device=self.device)
        
        return torch.tensor(0.0, device=self.device)
    
    def calculate_loss(
        self,
        reward: torch.Tensor,
        output: GeneratorOutput,
    ) -> torch.Tensor:
        """
        Calculate policy gradient loss.
        
        Loss = reward * (CE_start + CE_end)
        
        This is the REINFORCE loss that encourages actions leading to higher rewards.
        """
        ce_start = F.cross_entropy(
            output.p_start_logits.view(1, -1),
            output.a_start_idx.unsqueeze(0),
        )
        
        if output.a_end_idx == -1:
            # No-op selected, only penalize start selection
            return reward * ce_start
        
        ce_end = F.cross_entropy(
            output.p_end_logits.view(1, -1),
            output.a_end_idx.unsqueeze(0),
        )
        
        return reward * (ce_start + ce_end)
    
    def generate(
        self,
        num_steps: int | None = None,
        return_intermediate: bool = False,
    ) -> GeneratedGraph | list[GeneratedGraph]:
        """
        Generate a complete graph.
        
        Args:
            num_steps: Number of generation steps (default: max_gen_steps)
            return_intermediate: If True, return all intermediate graphs
            
        Returns:
            Final graph or list of intermediate graphs
        """
        if num_steps is None:
            num_steps = self.max_gen_steps
        
        self.eval()
        self.reset_graph()
        
        graphs = [self.current_graph.clone()] if return_intermediate else []
        
        with torch.no_grad():
            for _ in range(num_steps):
                output = self.forward(self.current_graph)
                reward = self._check_graph_rules(output.graph)
                
                if reward >= 0:
                    self.current_graph = output.graph
                    if return_intermediate:
                        graphs.append(self.current_graph.clone())
                
                # Check if no-op was intentionally selected at max nodes
                if output.a_end_idx == -1 and self.current_graph.num_nodes >= self.max_nodes:
                    break
        
        if return_intermediate:
            return graphs
        return self.current_graph
    
    def __repr__(self) -> str:
        return (
            f"Generator(target_class={self.target_class}, "
            f"max_nodes={self.max_nodes}, "
            f"candidates={[c.name for c in self.candidate_set]})"
        )
