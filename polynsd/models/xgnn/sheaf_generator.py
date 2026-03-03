from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, open_dict
from torch_geometric.data import Data
from polynsd.models.sheaf_gnn.transductive.disc_models import DiscreteSheafDiffusion
from polynsd.models.xgnn.generator import GeneratorOutput, CandidateNode

# import abstract for classes
from abc import ABC, abstractmethod

@dataclass
class SheafGeneratedGraph:
    """Container for generated graph state."""

    data: Data
    num_nodes: int
    degrees: torch.Tensor  # Degree of each node

    def clone(self) -> "SheafGeneratedGraph":
        """Create a deep copy of the graph."""
        return SheafGeneratedGraph(
            data=copy.deepcopy(self.data),
            num_nodes=self.num_nodes,
            degrees=self.degrees.clone(),
        )

    def to(self, device: torch.device) -> "SheafGeneratedGraph":
        """Move graph to device."""
        return SheafGeneratedGraph(
            data=self.data.to(device),
            num_nodes=self.num_nodes,
            degrees=self.degrees.to(device),
        )
class MLPEdgePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_edge_types: int):
        super().__init__()
        self.proj_start = nn.Linear(input_dim, hidden_dim)
        self.proj_end = nn.Linear(input_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim * 2, num_edge_types)
    
    def forward(
        self, x_start: torch.Tensor, x_end: torch.Tensor
    ) -> torch.Tensor:                                      # [1, input_dim]
        h_start = F.relu(self.proj_start(x_start))          # [1, hidden_dim]
        h_end = F.relu(self.proj_end(x_end))                # [1, hidden_dim]
        h = torch.cat([h_start, h_end], dim=-1)            # [1, hidden_dim * 2]
        return self.final(h).squeeze(0)                    # [num_edge_types]


class SheafGenerator(ABC, nn.Module):
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
        classifier: DiscreteSheafDiffusion,
        candidate_set: list[str],
        target_class: int = 0,
        num_classes: int = 2,
        max_nodes: int = 5,
        min_nodes: int = 2,
        max_gen_steps: int = 7,
        hyp_rollout: float = 1.0,
        hyp_rules: float = 2.0,
        num_rollouts: int = 10,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        num_edge_types: int = 1,
        num_node_types: int = 1,
        start_node_idx: int | None = None,
        device: torch.device | None = None,
        model_config: DictConfig | None = None,
        sheaf_settings: DictConfig | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dropout = dropout
        self.temperature = torch.Tensor([temperature]).to(self.device)
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.num_classes = num_classes
        self.max_gen_steps = max_gen_steps
        self.target_class = target_class
        # Parse candidate set
        self.candidate_set = [CandidateNode.from_string(c) for c in candidate_set]
        self.candidate_set_raw = candidate_set
        self.num_candidates = len(candidate_set)
        self.num_candidates_extended = self.num_candidates + 1  # +1 for no-op node

        # Hyperparameters
        self.hyp_rollout = hyp_rollout
        self.hyp_rules = hyp_rules
        self.num_rollouts = num_rollouts

        # Starting node configuration
        self.start_node_idx = start_node_idx
        self.random_start = start_node_idx is None

        # Number of node and edge types
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        # ------------------------------------------------------------------
        # Dynamic Sheaf Model Configuration

        if model_config is not None:
            # Create a copy to avoid modifying the original config
            self.model_config = copy.deepcopy(model_config)

            # Use open_dict context manager to allow adding keys to config if they are missing
            # and to allow modification if it is in struct mode
            with open_dict(self.model_config):
                # --- Override core dimensions for the generator ---
                # Input dim = number of node types (candidates + no-op)

                # We need to access args inside model_config
                if hasattr(self.model_config, "args"):
                    with open_dict(self.model_config.args):
                        self.model_config.args.input_dim = self.num_features
                        self.model_config.args.hidden_channels = hidden_dim
                        self.model_config.args.output_dim = hidden_dim
                        self.model_config.args.num_node_types = self.num_candidates_extended
                        self.model_config.args.num_edge_types = self.num_edge_types
                        self.model_config.args.device = str(self.device).split(":")[0]
                        self.model_config.args.dropout = dropout

                        # --- Apply specific overrides from generator.sheaf section ---
                        if sheaf_settings is not None:
                            # Apply top-level settings if any (usually args are nested)
                            for key, value in sheaf_settings.items():
                                if key == "transport":
                                    continue  # Handled separately
                                if key == "args":
                                    # Merge args
                                    for arg_key, arg_val in value.items():
                                        if arg_key != "_target_":
                                            setattr(
                                                self.model_config.args, arg_key, arg_val
                                            )
                                elif hasattr(self.model_config, key):
                                    setattr(self.model_config, key, value)
                                elif hasattr(self.model_config.args, key):
                                    setattr(self.model_config.args, key, value)
                else:
                    # Fallback if args is not present (unlikely for this codebase)
                    pass
            
            # Instantiate the model
            self.sheaf_model = hydra.utils.instantiate(self.model_config)


        else:
            raise ValueError("model_config must be provided to SheafGenerator")

        # Get actual stalk_dim from the created model
        stalk_dim = getattr(
            self.sheaf_model, "final_d", getattr(self.sheaf_model, "d", 1)
        )
        # The model's forward returns [N, d*channels]

        # ------------------------------------------------------------------
        start_dim = hidden_dim * max(stalk_dim, 1)  # Dimension of node embeddings used for start node selection
        # MLP for selecting start node
        self.mlp_start = nn.Sequential(
            nn.Linear(start_dim, 16), # TODO: make as hyperparameter (?)
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # MLP for selecting end node (takes concatenated start and candidate embeddings)
        self.mlp_end = nn.Sequential(       # input Nxfeature_dim *2 
            nn.Linear(start_dim * 2, 24),   # Nx24
            nn.ReLU(), 
            nn.Linear(24, 1),               # Nx1
        )

        # MLP for selecting edge type if multiple edge types are used 
        if self.num_edge_types > 1: 
            self.mlp_edge = MLPEdgePredictor(start_dim, 32, self.num_edge_types)
        else:
            self.mlp_edge = None

        # TODO:
        # instead of using a MLP, we could use the already trained sheaf model and classify the current graph
        # with all edge types and see which one increases the target class probability the most
        # Does this increase the computational cost too much?
        
        # Store the classifier (frozen)
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        
        # Initialize tracking
        self.gen_steps_done = 0
        self.current_graph: SheafGeneratedGraph | None = None
        
        # Move to device
        self.to(self.device)
        

    def reset_graph(self) -> SheafGeneratedGraph:
        """Reset to initial graph with single starting node."""
        self.gen_steps_done = 0
        
        # Select starting node
        if self.random_start:
            start_idx = random.randint(0, self.num_candidates - 1)
        else:
            start_idx = self.start_node_idx
        
        # # Create one-hot feature for starting node
        # feat = torch.zeros((1, self.num_features), dtype=torch.float32, device=self.device)
        # feat[0, start_idx] = 1.0
        
        # # Empty edge index
        # edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        feat = torch.ones((1, self.num_features), dtype=torch.float32, device=self.device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Initial degree
        degrees = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        new_data = Data(
            x=feat,
            edge_index=edge_index,
            batch=torch.zeros(1, dtype=torch.long, device=self.device),
        )
        new_data.node_type = torch.tensor([start_idx], dtype=torch.long, device=self.device)
        new_data.num_node_types = self.num_candidates + 1  # candidates + no-op

        if self.num_edge_types > 1:
            new_data.edge_type = torch.empty((0,), dtype=torch.long, device=self.device)
        else:
            new_data._store.__dict__["edge_type"] = None
            new_data.edge_type = None

        new_data.num_edge_types = self.num_edge_types


        self.current_graph = SheafGeneratedGraph(
            data=new_data,
            num_nodes=1,
            degrees=degrees,
        )
        
        return self.current_graph
    
    def _compute_node_embeddings(
        self,
        graph: SheafGeneratedGraph,
    ) -> torch.Tensor:
        """
        Compute node embeddings for graph nodes and candidate nodes.
        
        Returns embeddings for: [graph_nodes, candidate_nodes, no_op_node]
        """
        graph_data = graph.data

        #print(f"graph_data.x shape: {graph_data.x.shape}, edge_index shape: {graph_data.edge_index.shape}", flush=True)
        #print(f"graph_data.node_type: {graph_data.node_type}", flush=True)
        #print(f"graph_data.x: {graph_data.x}", flush=True)
            
        x_new, node_type_new = self._get_candidates(graph_data)
        x = torch.cat([graph_data.x, x_new], dim=0)
        edge_index = graph_data.edge_index
        node_type = torch.cat([graph_data.node_type, node_type_new], dim=0)


        new_data = Data(x, edge_index, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))  
        new_data.node_type = node_type
        new_data.num_node_types = self.num_candidates_extended  # candidates + no-op
        new_data._store.__dict__["edge_type"] = graph_data._store.__dict__.get("edge_type", None)  # Preserve edge_type if it exists
        new_data.edge_type = graph_data.edge_type  # Preserve edge_type if it exists
        new_data.num_edge_types = self.num_edge_types  # Set to generator's edge types 

        # Update encoder for this graph's topology (inductive).
        num_nodes = new_data.x.size(0)
        if hasattr(self.sheaf_model, "graph_size"):
            self.sheaf_model.graph_size = num_nodes
        if hasattr(self.sheaf_model, "edge_index"):
            self.sheaf_model.edge_index = new_data.edge_index
        # Update builders if present.
        self.sheaf_model.regenerate_builder(num_nodes, new_data.edge_index)


        node_embeddings, _ = self.sheaf_model(new_data) # we return a tensor with stalks

        return node_embeddings, x_new
    
    def forward(self, graph: SheafGeneratedGraph) -> GeneratorOutput:
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
        node_embeddings, x_candidate_new = self._compute_node_embeddings(G)
        
        num_total = G.num_nodes + self.num_candidates * self.samples_per_candidate + 1  # +1 for no-op

        # Create mask for start node selection
        # Only existing graph nodes can be start nodes (not candidates, not no-op)
        # But we allow no-op to be selected (last position)
        mask_start = torch.zeros(num_total, dtype=torch.bool, device=self.device)
        mask_start[G.num_nodes:num_total - 1] = True  # Mask candidates
        mask_start = mask_start.view(-1, 1)
        
        # Compute start node probabilities
        p_start_logits = self.mlp_start(node_embeddings) # MLP on: actual graph + [candidates * samples_per_candidate + no-op]
        p_start = F.softmax(p_start_logits.masked_fill(mask_start, -1e8) / self.temperature, dim=0)
        
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
        emb_start = node_embeddings[a_start_idx].unsqueeze(0).expand(num_total, -1) # [1, feature_dim] to [total_nodes, feature_dim]
        emb_combined = torch.cat([node_embeddings, emb_start], dim=1) # [total_nodes, feature_dim*2], i.e. concatenation is done on columns

        # Create mask for end node selection
        # Cannot select: start node itself, no-op node
        mask_end = torch.zeros(num_total, dtype=torch.bool, device=self.device)
        mask_end[a_start_idx] = True  # Can't connect to self
        mask_end[-1] = True  # Can't select no-op as end
        mask_end = mask_end.view(-1, 1)
        
        # Compute end node probabilities
        p_end_logits = self.mlp_end(emb_combined) # logits return [total_nodes, 1]
        p_end = F.softmax(p_end_logits.masked_fill(mask_end, -1e8) / self.temperature, dim=0)
        
        # Sample end node
        a_end_idx = torch.multinomial(p_end.view(-1), num_samples=1).squeeze()

        x_end_new = x_candidate_new[a_end_idx - G.num_nodes] if a_end_idx >= G.num_nodes else G.data.x[a_end_idx]  # Get feature for new node if candidate, else existing node
        x_end_new = x_end_new.unsqueeze(0)  # [1, feature_dim]
        # Update graph
        G = self._add_edge(G, a_start_idx.item(), a_end_idx.item(), x_end_new, node_embeddings)
        
        return GeneratorOutput(
            p_start_logits=p_start_logits,
            a_start_idx=a_start_idx,
            p_end_logits=p_end_logits,
            a_end_idx=a_end_idx,
            graph=G,
        )
    
    def _add_edge(
        self,
        graph: SheafGeneratedGraph,
        start_idx: int,
        end_idx: int,
        x_end: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> SheafGeneratedGraph:
        """Add edge to graph, potentially adding a new node."""
        G = graph.clone()
        
        # Create new node feature
        emb_start = node_embeddings[start_idx].unsqueeze(0)  # [1, feature_dim]
        emb_end = node_embeddings[end_idx].unsqueeze(0) # [1, feature_dim]
        edge_type = None

        if end_idx >= G.num_nodes:
            # End node is from candidate set - add new node
            
            # Get the candidate index (accounting for existing nodes)
            candidate_idx = end_idx - G.num_nodes
            actual_type = candidate_idx % self.num_candidates  # Wrap around if using multiple candidates per iteration #TODO: giusto?  kri dice di si, speriamo bene (ho sbollato) [SOS]
            node_type = torch.tensor([actual_type], dtype=torch.long, device=self.device)
            
            # Add to feature matrix
            new_feat = x_end.to(self.device)  # Use the embedding of the selected candidate as the new node feature
            G.data.x = torch.cat([G.data.x, new_feat], dim=0)
            G.data.node_type = torch.cat([G.data.node_type, node_type], dim=0)

            if G.data.edge_type is not None:  # it can be None if dataset is homogeneous and we are not using edge types, but if we are using edge types we need to add them for the new edges
                if self.num_edge_types > 1:
                    edge_type_logits = self.mlp_edge(emb_start, emb_end)
                    mask_edge_type = self._get_mask_edge_type(G.data.node_type[start_idx], actual_type)
                    edge_type = F.softmax(edge_type_logits.masked_fill(mask_edge_type, -1e8) / self.temperature, dim=0).argmax().item()

                    G.data.edge_type = torch.cat([G.data.edge_type, torch.tensor([edge_type, edge_type], dtype=torch.long, device=self.device)], dim=0)
                
                else: #we use edge types bc we are hetrogeneous, but we only have 1 edge type, so we need to add it for the new edges
                    # Single edge type — still need to keep edge_type in sync
                    G.data.edge_type = torch.cat([G.data.edge_type, torch.zeros(2, dtype=torch.long, device=self.device)], dim=0)
                    edge_type = 0  # Default edge type if not using edge prediction
                    
            # Add edges (bidirectional)
            new_edges = torch.tensor(
                [[start_idx, G.num_nodes], [G.num_nodes, start_idx]],
                dtype=torch.long,
                device=self.device,
            )
            G.data.edge_index = torch.cat([G.data.edge_index, new_edges], dim=1)
            
            # Update degrees
            self._update_edge_degree(G, start_idx, end_idx, edge_type, is_new_node=True)
            
            # Update node count
            G.num_nodes += 1
        else:  # End node exists - just add edge
            
            if  G.data.edge_type is not None:  # Only predict edge type if we are tracking edge types
                if self.num_edge_types > 1:
                    edge_type_logits = self.mlp_edge(emb_start, emb_end)
                    mask_edge_type = self._get_mask_edge_type(G.data.node_type[start_idx], G.data.node_type[end_idx])
                    edge_type = F.softmax(edge_type_logits.masked_fill(mask_edge_type, -1e8) / self.temperature, dim=0).argmax().item()

                else:
                    edge_type = 0  # Default edge type if not using edge prediction

                G.data.edge_type = torch.cat([G.data.edge_type, torch.tensor([edge_type, edge_type], dtype=torch.long, device=self.device)], dim=0)

            new_edges = torch.tensor(
                [[start_idx, end_idx], [end_idx, start_idx]],
                dtype=torch.long,
                device=self.device,
            )
            G.data.edge_index = torch.cat([G.data.edge_index, new_edges], dim=1)
            
            # Update degrees
            self._update_edge_degree(G, start_idx, end_idx, edge_type, is_new_node=False)
        
        return G

    def _update_edge_degree(
        self,
        graph: SheafGeneratedGraph,
        start_idx: int,
        end_idx: int,
        edge_type: int,
        is_new_node: bool,
    ) -> None:
        """Update degree information for nodes after adding an edge."""

        update_value = 1 
        graph.degrees[start_idx] += update_value
        if not is_new_node:
            graph.degrees[end_idx] += update_value
        else:
            graph.degrees = torch.cat([graph.degrees, torch.tensor([update_value], dtype=torch.float32, device=self.device)], dim=0)
                    

    @abstractmethod
    def _get_mask_edge_type(
        self,
        start_node_type: int,
        end_node_type: int,
    ) -> torch.Tensor: ...

    
    def calculate_reward(
        self,
        graph: SheafGeneratedGraph,
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


        r_edge = 0
        # reward higher class edge_type
        if graph.data.edge_type is not None and self.num_edge_types > 1 and graph.data.edge_type.numel() > 0:
            is_not_single = graph.data.edge_type[-1].float() != 1 # Reward based on the type of the last added edge
            r_edge = graph.data.edge_type[-1].float() + 1  if is_not_single else 0
            #print(", reward edge_type:", r_edge, flush=True)
        
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
        
        # we return reward and validity (reward >= 0 means valid graph according to rules)
        return r_feedback + self.hyp_rules * r_rules + r_edge, r_rules >= 0  
    
    def _calculate_classifier_reward(self, graph: SheafGeneratedGraph) -> torch.Tensor:
        """
        Calculate reward from classifier predictions.
        
        Reward = P(target_class | graph) - 1/num_classes
        
        This encourages graphs that increase confidence for target class
        above random chance.
        """
        self.classifier.eval()
        with torch.no_grad():
            probs = self.get_probs(graph.clone(), self.classifier)
        
        return probs[0, self.target_class] - (1.0 / self.num_classes)
    
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
    ) -> SheafGeneratedGraph | list[SheafGeneratedGraph]:
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
        
        with torch.no_grad(): #TODO: check here and visualization for edge types ;-) 
            for _ in range(num_steps):
                output = self.forward(self.current_graph)
                reward = self._check_graph_rules(output.graph)
                
                print(f"Step {_+1}/{num_steps}, Reward: {reward.item():.4f}", flush=True)
                # Handle a_start_idx and a_end_idx which can be tensors or integers
                start_idx = output.a_start_idx.item() if isinstance(output.a_start_idx, torch.Tensor) else output.a_start_idx
                end_idx = output.a_end_idx.item() if isinstance(output.a_end_idx, torch.Tensor) else output.a_end_idx
                print(f"Output action: start_idx={start_idx}, end_idx={end_idx}", flush=True)
                print(f"Output p logits: start={output.p_start_logits.squeeze().cpu().numpy()}, end={output.p_end_logits.squeeze().cpu().numpy() if output.p_end_logits is not None else None}", flush=True)

                # zip node types and p_start_logits for debugging
                node_types = output.graph.data.node_type.cpu().numpy()
                p_start_logits = output.p_start_logits.squeeze().cpu().numpy()
                print("Node types and start logits:")
                for nt, logit in zip(node_types, p_start_logits):
                    nt_str = self.candidate_set_raw[nt] if nt < self.num_candidates else "NO-OP"
                    print(f"  Node type: {nt_str} (idx {nt}), Start logit: {logit:.4f}")

                if reward >= 0:
                    self.current_graph = output.graph
                    if return_intermediate:
                        graphs.append(self.current_graph.clone())
                
                # Check if no-op was intentionally selected at max nodes
                if output.a_end_idx == -1 and self.current_graph.num_nodes >= self.max_nodes:
                    print("No-op selected at max nodes, stopping generation.", flush=True)
                    break
        
        if return_intermediate:
            return graphs
        return self.current_graph

    def get_probs(
        self,
        graph: SheafGeneratedGraph,
        graph_classifier: DiscreteSheafDiffusion) -> torch.Tensor:
        """
        Get classifier probabilities for the generated graph.
        Args:
            graph: Generated graph
            graph_classifier: Classifier to evaluate
        Returns:
            Class probabilities tensor
        """

        G = graph.clone().to(self.device)
        
        mask = G.data.node_type != self.num_candidates  # Keep nodes that are not no-op
        G.data.x = G.data.x[mask]
        #G.data.x = G.data.x[:, :-1] #TODO controllare che faceva questo? giusto?
        G.data.node_type = G.data.node_type[mask]
        # Filter edges to keep only those connecting non-no-op nodes
        edge_mask = mask[G.data.edge_index[0]] & mask[G.data.edge_index[1]]
        G.data.edge_index = G.data.edge_index[:, edge_mask]

        G.data.batch = torch.zeros(G.data.x.size(0), dtype=torch.long, device=G.data.x.device)
        num_edges = G.data.edge_index.size(1) if G.data.edge_index.numel() > 0 else 0
        G.data.edge_type = torch.zeros(num_edges, dtype=torch.long, device=G.data.x.device)
        
        #print(G.data.x)
        #print(G.data.edge_index, flush=True)

        graph_classifier.encoder.regenerate_builder(G.num_nodes, G.data.edge_index)
        probs = graph_classifier.predict_proba(G.data)
        return probs
    
    def __repr__(self) -> str:
        return (
            f"Generator(target_class={self.target_class}, "
            f"max_nodes={self.max_nodes}, "
            f"candidates={[c.name for c in self.candidate_set]})"
        )

    @abstractmethod
    def _check_graph_rules(self, graph: SheafGeneratedGraph) -> torch.Tensor: ...

    @abstractmethod
    def _get_candidates(self, graph_data: Data) -> tuple[torch.Tensor, torch.Tensor]: ...

    def __str__(self) -> str:
        return "SheafGenerator" 
    
class SheafGeneratorMUTAG(SheafGenerator):
    """SheafGenerator with Mutag-specific rules."""
    def __init__(
        self,
        classifier: DiscreteSheafDiffusion,
        num_classes: int = 2,
        max_nodes: int = 5,
        min_nodes: int = 2,
        max_gen_steps: int = 7,
        hyp_rollout: float = 1.0,
        hyp_rules: float = 2.0,
        num_rollouts: int = 10,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        num_node_types: int = 1,
        num_edge_types: int = 1,
        start_node_idx: int | None = None,
        device: torch.device | None = None,
        model_config: DictConfig | None = None,
        sheaf_settings: DictConfig | None = None,
        temperature: float = 1.0,
    ):
        target_class = 0
        self.num_features = 1  
        self.samples_per_candidate = 1
        print("num_edge_types:", num_edge_types, flush=True)
        print("num_node_types:", num_node_types, flush=True)

        candidate_set = ["C.4", "N.3", "O.2", "F.1", "Cl.1", "Br.1", "I.1"]  # Default MUTAG candidates
        super().__init__(
            classifier=classifier,
            candidate_set=candidate_set,
            target_class=target_class,
            num_classes=num_classes,
            max_nodes=max_nodes,
            min_nodes=min_nodes,
            max_gen_steps=max_gen_steps,
            hyp_rollout=hyp_rollout,
            hyp_rules=hyp_rules,
            num_rollouts=num_rollouts,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            start_node_idx=start_node_idx,
            device=device,
            model_config=model_config,
            sheaf_settings=sheaf_settings,
            temperature=temperature,
        )

    @override     
    def __str__(self) -> str:
        return "SheafGeneratorMUTAG" 

    @override
    def _get_candidates(self, graph_data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        # For MUTAG, candidates are just the predefined set, not dependent on graph state
        x_new = torch.ones((self.num_candidates_extended, self.num_features), device=graph_data.x.device, dtype=graph_data.x.dtype)
        node_type_new = torch.arange(self.num_candidates_extended, device=graph_data.x.device, dtype=torch.long)
        return x_new, node_type_new
    
    @override
    def _get_mask_edge_type(
        self,
        start_node_type: int,
        end_node_type: int,
    ) -> torch.Tensor:
        """Mask/select MUTAG bond type based on atom types and (approx.) valency.

        MUTAG edge types (aromatic, single, double, triple) -> indices (0, 1, 2, 3).

        This function:
        - Disallows aromatic/double/triple for halogens.
        - Disallows bond orders that exceed min(atom valencies) for the end node.
        """
        start_valency = self.candidate_set[start_node_type].max_valency
        end_valency = self.candidate_set[end_node_type].max_valency

        #order is (aromatic, single, double, triple), so both aromatic and single are allowed.
        bond = min(start_valency, end_valency) # Max possible bond order based on valency 
        
        # mask all edge types that exceed the allowed bond order
        mask_edge = torch.zeros(self.num_edge_types, dtype=torch.bool, device=self.device)
        mask_edge[bond+1:] = True  # Mask bond orders that are too high for the given node types

        # we can have aromatic only if both nodes have valency >= 2: 
        # only bonds between {C, N, O} can be aromatic.
        if bond < 2 or True: #TODO: we mask aromatic for now and keep the aromatic as implicit.
            mask_edge[0] = True  # Mask aromatic for single bonds

        return mask_edge
    
    @override
    def _update_edge_degree(
        self,
        graph: SheafGeneratedGraph,
        start_idx: int,
        end_idx: int,
        edge_type: int,
        is_new_node: bool,
    ) -> None:
        """Update degree information for nodes after adding an edge."""

        update_value = 1.5
        if edge_type != 0: 
            update_value = edge_type  # Example: edge_type 1 adds 1, edge_type 2 adds 2, edge_type 3 adds 3
        graph.degrees[start_idx] += update_value
        if not is_new_node:
            graph.degrees[end_idx] += update_value
        else:
            graph.degrees = torch.cat([graph.degrees, torch.tensor([update_value], dtype=torch.float32, device=self.device)], dim=0)
                    
    @override
    def _check_graph_rules(self, graph: SheafGeneratedGraph) -> torch.Tensor:
        """
        Check if graph satisfies domain rules.
        
        Rules:
        1. No duplicate edges
        2. Node degrees don't exceed valency
        3. No premature no-op (when under max nodes)
        4. Don't exceed max nodes
        5. Reward higher edge types (optional)
        
        Returns:
            Reward tensor (negative if invalid, positive for desirable properties)
        """

        is_noop_selected = (self.current_graph is not None) and (graph.data.edge_index.shape[1] == self.current_graph.data.edge_index.shape[1]) and (graph.num_nodes == self.current_graph.num_nodes)

        # Check valency constraints
        for idx, degree in enumerate(graph.degrees):
            if degree > 0:
                #node_type_idx = graph.feat[idx].argmax().item()
                node_type_idx = graph.data.node_type[idx].item()
                if node_type_idx < len(self.candidate_set):
                    max_valency = self.candidate_set[node_type_idx].max_valency
                    if degree > max_valency:
                        return torch.tensor(-1.0, device=self.device)
                    
                    #if is_noop_selected and degree % 1 != 0: 
                        # No-op selected but degree is not an integer, 
                        # which means we have a half bond (aromatic) that cannot be formed without adding a new node, 
                        # so this is invalid
                        # (otherwise there can be H implicitly added to satisfy valency)
                    #    return torch.tensor(-1.0, device=self.device)
        
        # Check for duplicate edges
        edges = graph.data.edge_index.t().tolist()
        edge_set = {tuple(e) for e in edges}
        if len(edge_set) < len(edges):
            return torch.tensor(-1.0, device=self.device)
        
        # Check for premature no-op
        #if is_noop_selected and graph.num_nodes < self.min_nodes:
            # i.e. no-op selected before reaching max nodes, which is not allowed
            #return torch.tensor(-1.0, device=self.device)

        if is_noop_selected and graph.num_nodes < self.min_nodes:
            return torch.tensor(-(self.min_nodes - graph.num_nodes), device=self.device)  # Penalize based on how many more nodes are needed to reach min_nodes

        # if is_noop_selected: 
        #     return torch.tensor(-1.0, device=self.device)

        # Check max nodes
        if graph.num_nodes > self.max_nodes:
            return torch.tensor(-1.0, device=self.device)
        
        # Reward higher edge types
        reward = torch.tensor(0.0, device=self.device)
        # reward += graph.degrees.sum() * 0.1  # Example: reward based on total degree (more edges can be better)
        # reward += graph.num_nodes * 0.2  # Example: reward based on number of nodes (encourages growth)

        # if graph.data.edge_type is not None and self.num_edge_types > 1:
        #     avg_edge_type = graph.data.edge_type.float().mean()
        #     reward += avg_edge_type * 0.3  # Scale factor to control reward magnitude
        
        return reward

class SheafGeneratorPROTEINS(SheafGenerator):
    """SheafGenerator with PROTEINS-specific rules."""

    def __init__(
        self,
        classifier: DiscreteSheafDiffusion,
        num_classes: int = 2,
        max_nodes: int = 5,
        min_nodes: int = 2,
        max_gen_steps: int = 7,
        hyp_rollout: float = 1.0,
        hyp_rules: float = 2.0,
        num_rollouts: int = 10,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        num_node_types: int = 1,
        num_edge_types: int = 1,
        start_node_idx: int | None = None,
        device: torch.device | None = None,
        model_config: DictConfig | None = None,
        sheaf_settings: DictConfig | None = None,
        temperature: float = 1.0,
    ):
        
        candidate_set = ["hlx", "sht", "loop"]  # Default PROTEINS candidates
        target_class = 1 
        self.num_features = 1
        self.samples_per_candidate = 3 #TODO: li passiamo nel costruttore per i file di config?

        super().__init__(
            classifier=classifier,
            candidate_set=candidate_set,
            target_class=target_class,
            num_classes=num_classes,
            max_nodes=max_nodes,
            min_nodes=min_nodes,
            max_gen_steps=max_gen_steps,
            hyp_rollout=hyp_rollout,
            hyp_rules=hyp_rules,
            num_rollouts=num_rollouts,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            start_node_idx=start_node_idx,
            device=device,
            model_config=model_config,
            sheaf_settings=sheaf_settings,
            temperature=temperature,
        )
        print("num_features:", self.num_features, flush=True)
        print("candidate_set:", self.candidate_set, flush=True)
        print("num_candidates:", self.num_candidates, flush=True)
        print("num_candidates_extended:", self.num_candidates_extended, flush=True)
        print("num_node_types:", self.num_node_types, flush=True)
        print("num_edge_types:", self.num_edge_types, flush=True)
        # Initialize distribution on the target device
        self.dist = torch.distributions.Pareto(scale=torch.tensor([1.0], device=device), alpha=torch.tensor([2.5], device=device))

    @override
    def __str__(self) -> str:
        return "SheafGeneratorPROTEINS"

    def _get_candidates(self, graph_data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        # For PROTEINS, we generate candidates with random features to encourage exploration, since the space of possible connections is more complex and not easily enumerable.
        x_new = torch.empty((0, self.num_features), device=graph_data.x.device)
        node_type_new = torch.empty((0,), dtype=torch.long, device=graph_data.x.device)

        for _ in range(self.samples_per_candidate):
            x_candidate = torch.ones((self.num_candidates, self.num_features), device=graph_data.x.device)
            x_candidate += self.dist.sample((self.num_candidates, self.num_features)).squeeze(-1).to(graph_data.x.device)
            
            x_new = torch.cat([x_new, x_candidate], dim=0)
            node_type_new = torch.cat([node_type_new, torch.arange(self.num_candidates, device=graph_data.x.device)], dim=0) #arange goes from 0 to num_candidates-1, which is what we want for node types of candidates
        
        
        #print("x_new shape", x_new.shape, flush=True) 3x3=9 works!
        # add no-op candidate
        x_new = torch.cat([x_new, torch.ones((1, self.num_features), device=graph_data.x.device)], dim=0)
        node_type_new = torch.cat([node_type_new, torch.tensor([self.num_candidates], device=graph_data.x.device)], dim=0) # no-op candidate has node type equal to num_candidates (the last index)

        return x_new, node_type_new

    @override
    def _check_graph_rules(self, graph: SheafGeneratedGraph) -> torch.Tensor:
        # often there is a loop (turn) between helixs and sheets, even if its physically possible that they are connected
        # note that there are sequential edges (helix-turn-sheet-turn-...) and spatial edges (helix-sheet, helix-helix, sheet-sheet)
        # (helix-sheet) sequential edges are very rare 

        # Connection                Sequential (Chain)          Spatial (3D Proximity)
        # Sheet to Sheet            Common                      "Constant (to form the ""Sheet"")"
        # Helix to Helix            Common                      Very Common (Bundles)
        # Helix to Sheet            Rare (usually bridged)      Very Common
        # Any to Turn               Universal                   Common
        #
        #   TODO: maybe add a rule to penalize them, but for now we just ignore them

        # Max 2 nodes have degree = 1 (end of chains)
        degree_one_count = (graph.degrees == 1).sum().item()
        if degree_one_count > 2:
            return torch.tensor(-1.0, device=self.device)

        # Check for duplicate edges
        edges = graph.data.edge_index.t().tolist()
        edge_set = {tuple(e) for e in edges}
        if len(edge_set) < len(edges):
            return torch.tensor(-1.0, device=self.device)
        
        # Check for premature no-op
        if self.current_graph is not None:
            if (graph.num_nodes == self.current_graph.num_nodes and
                graph.num_nodes < self.min_nodes and
                graph.data.edge_index.shape[1] == self.current_graph.data.edge_index.shape[1]):
                return torch.tensor(-1.0, device=self.device)
        
        # Check max nodes
        if graph.num_nodes > self.max_nodes:
            return torch.tensor(-1.0, device=self.device)
        
        return torch.tensor(0.0, device=self.device)  # Placeholder for valid graph