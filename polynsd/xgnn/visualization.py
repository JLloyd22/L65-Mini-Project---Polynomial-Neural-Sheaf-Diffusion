#  Copyright (c) 2024 
#  Adapted from X-GNN: Model-Explanations of GNNs using RL
#  License: CC0 1.0 Universal (CC0 1.0)
"""
Visualization utilities for XGNN generated graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from polynsd.models.xgnn.generator import GeneratedGraph, Generator
    from polynsd.models.xgnn.sheaf_generator import SheafGeneratedGraph


# Default atom colors for molecular graphs (MUTAG) # TODO: change colors
ATOM_COLORS = {
    'C': 'grey',
    'N': 'lightblue',
    'O': 'red',
    'F': 'yellow',
    'I': 'magenta',
    'Cl': 'lightgreen',
    'Br': 'orange',
} 


def visualize_sheaf_generated_graph(
    generator: "Generator",
    graph: "SheafGeneratedGraph | None" = None,
    candidate_set: list[str] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    node_size: int = 500,
    font_size: int = 10,
    with_labels: bool = True,
    title: str | None = None,
    probs: list[float] | None = None,
) -> plt.Figure | None:
    """
    Visualize a sheaf generated graph.
    
    Args:
        generator: Generator instance (used for classifier predictions)
        graph: Graph to visualize (default: generator's current graph)
        candidate_set: List of candidate node types (e.g., ['C.4', 'N.3'])
        ax: Matplotlib axes to plot on
        show: Whether to call plt.show()
        node_size: Size of nodes in visualization
        font_size: Font size for labels
        with_labels: Whether to show node labels
        title: Plot title

    Returns:
        Figure if show=False, None otherwise
    """
    if graph is None:
        graph = generator.current_graph
    
    if candidate_set is None:
        candidate_set = generator.candidate_set_raw
    
    # Create NetworkX graph
    G_nx = nx.Graph()
    
    # Add nodes
    for i in range(graph.num_nodes):
        G_nx.add_node(i)
    
    # Add edges (accounting for bidirectional representation)
    edges = graph.data.edge_index.t().tolist()
    edge_types = getattr(graph.data, "edge_type", None)
    edge_types_list: list[int] | None = None
    if edge_types is not None:
        # edge_type is expected to be aligned with edge_index columns
        edge_types_list = edge_types.detach().cpu().view(-1).tolist()
    edge_count = {}
    edge_type_map: dict[tuple[int, int], set[int]] = {}
    for col, (u, v) in enumerate(edges):
        edge = tuple(sorted((u, v)))
        edge_count[edge] = edge_count.get(edge, 0) + 1
        if edge_types_list is not None and col < len(edge_types_list):
            edge_type_map.setdefault(edge, set()).add(int(edge_types_list[col]))
    
    for (u, v), count in edge_count.items():
        G_nx.add_edge(u, v, weight=count / 2)
    
    # Get layout
    pos = nx.spring_layout(G_nx, seed=42)
    
    # Get node types and colors
    node_types = graph.data.node_type.cpu().numpy()
    
    # Create figure if needed
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Parse candidate set for labels and colors
    atom_names = [c.split('.')[0] for c in candidate_set]
    
    # Draw nodes by type
    for i, atom_name in enumerate(atom_names):
        nodes = [n for n in G_nx.nodes() if node_types[n] == i]
        if nodes:
            color = ATOM_COLORS.get(atom_name, 'gray')
            nx.draw_networkx_nodes(
                G_nx, pos, nodelist=nodes,
                node_color=color, node_size=node_size, ax=ax,
            )
            if with_labels:
                labels = {n: atom_name for n in nodes}
                nx.draw_networkx_labels(
                    G_nx, pos, labels=labels,
                    font_size=font_size, ax=ax,
                )

    # Draw edges styled as bonds based on edge type (0: dotted, 1: single, 2: double, 3: triple)
    # We draw parallel curved edges for double/triple bonds using connectionstyle arcs.
    def _draw_bond(u, v, bond_order: int):
        if bond_order == 0:
            # dotted single line
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], style='dotted', edge_color='gray', ax=ax, width=1.2, arrows=True)
        elif bond_order == 1:
            # single solid line
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, width=2.0)
        elif bond_order == 2:
            # double bond: two parallel curved lines
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.08', width=1.6, arrows=True)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=-0.08', width=1.6, arrows=True)
        else:
            # triple bond: three parallel lines (center + two arcs)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.12', width=1.2, arrows=True)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.0', width=2.0, arrows=True)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=-0.12', width=1.2, arrows=True)

    for u, v in G_nx.edges():
        weight = G_nx[u][v].get('weight', 1)
        edge = tuple(sorted((u, v)))
        # Prefer explicit edge type if available, otherwise fall back to multiplicity/weight
        if edge in edge_type_map and edge_type_map[edge]:
            bond_order = max(edge_type_map[edge])
        else:
            # fallback: use multiplicity from weight (round and cap between 0 and 3)
            bond_order = int(min(3, max(0, round(weight))))
        _draw_bond(u, v, bond_order)
        
    # Format title
    if title is None:
        if probs is None:
            title = "Generated Graph"
        else:
            prob_str = ", ".join([f"c{i}={float(p):.3f}" for i, p in enumerate(probs)])
            title = f"Generated Graph\nPredictions: {prob_str}"
    ax.set_title(title)
    ax.axis('off') 
    if show:
        plt.tight_layout()
        plt.show()
        return None
    return fig

def visualize_generated_graph(
    generator: "Generator",
    graph: "GeneratedGraph | None" = None,
    candidate_set: list[str] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    node_size: int = 500,
    font_size: int = 10,
    with_labels: bool = True,
    title: str | None = None,
) -> plt.Figure | None:
    """
    Visualize a generated graph.
    
    Args:
        generator: Generator instance (used for classifier predictions)
        graph: Graph to visualize (default: generator's current graph)
        candidate_set: List of candidate node types (e.g., ['C.4', 'N.3'])
        ax: Matplotlib axes to plot on
        show: Whether to call plt.show()
        node_size: Size of nodes in visualization
        font_size: Font size for labels
        with_labels: Whether to show node labels
        title: Plot title
        
    Returns:
        Figure if show=False, None otherwise
    """
    if graph is None:
        graph = generator.current_graph
    
    if candidate_set is None:
        candidate_set = generator.candidate_set_raw
    
    # Create NetworkX graph
    G_nx = nx.Graph()
    
    # Add nodes
    for i in range(graph.num_nodes):
        G_nx.add_node(i)
    
    # Add edges (accounting for bidirectional representation)
    edges = graph.edge_index.t().tolist()
    edge_types = getattr(graph, "edge_type", None)
    edge_types_list: list[int] | None = None
    if edge_types is not None:
        edge_types_list = edge_types.detach().cpu().view(-1).tolist()
    edge_count = {}
    edge_type_map: dict[tuple[int, int], set[int]] = {}
    for col, (u, v) in enumerate(edges):
        edge = tuple(sorted((u, v)))
        edge_count[edge] = edge_count.get(edge, 0) + 1
        if edge_types_list is not None and col < len(edge_types_list):
            edge_type_map.setdefault(edge, set()).add(int(edge_types_list[col]))
    
    for (u, v), count in edge_count.items():
        G_nx.add_edge(u, v, weight=count / 2)
    
    # Get layout
    pos = nx.spring_layout(G_nx, seed=42)
    
    # Get node types and colors
    node_types = torch.argmax(graph.feat, dim=1).cpu().numpy()
    
    # Create figure if needed
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Parse candidate set for labels and colors
    atom_names = [c.split('.')[0] for c in candidate_set]
    
    # Draw nodes by type
    for i, atom_name in enumerate(atom_names):
        nodes = [n for n in G_nx.nodes() if node_types[n] == i]
        if nodes:
            color = ATOM_COLORS.get(atom_name, 'gray')
            nx.draw_networkx_nodes(
                G_nx, pos, nodelist=nodes,
                node_color=color, node_size=node_size, ax=ax,
            )
            if with_labels:
                labels = {n: atom_name for n in nodes}
                nx.draw_networkx_labels(
                    G_nx, pos, labels=labels,
                    font_size=font_size, ax=ax,
                )
    
    # Draw edges styled as bonds based on edge type (0: dotted, 1: single, 2: double, 3: triple)
    # We draw parallel curved edges for double/triple bonds using connectionstyle arcs.
    def _draw_bond(u, v, bond_order: int):
        if bond_order == 0:
            # dotted single line
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], style='dotted', edge_color='gray', ax=ax, width=1.2, arrows=True)
        elif bond_order == 1:
            # single solid line
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, width=2.0, arrows=True)
        elif bond_order == 2:
            # double bond: two parallel curved lines
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.08', width=1.6, arrows=True)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=-0.08', width=1.6, arrows=True)
        else:
            # triple bond: three parallel lines (center + two arcs)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.12', width=1.2, arrows=True)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=0.0', width=2.0, arrows=True)
            nx.draw_networkx_edges(G_nx, pos, edgelist=[(u, v)], edge_color='gray', ax=ax, connectionstyle='arc3,rad=-0.12', width=1.2, arrows=True)

    for u, v in G_nx.edges():
        weight = G_nx[u][v].get('weight', 1)
        edge = tuple(sorted((u, v)))
        # Prefer explicit edge type if available, otherwise fall back to multiplicity/weight
        if edge in edge_type_map and edge_type_map[edge]:
            bond_order = max(edge_type_map[edge])
        else:
            # fallback: use multiplicity from weight (round and cap between 0 and 3)
            bond_order = int(min(3, max(0, round(weight))))
        _draw_bond(u, v, bond_order)
    
    # Get classifier predictions
    generator.classifier.eval()
    with torch.no_grad():
        x = graph.feat[:, :-1].to(generator.device)
        edge_index = graph.edge_index.to(generator.device)
        # Create Data object for classifier
        num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0
        data = Data(
            x=x, 
            edge_index=edge_index, 
            batch=torch.zeros(graph.num_nodes, dtype=torch.long, device=generator.device),
            edge_type=torch.zeros(num_edges, dtype=torch.long, device=generator.device),
            node_type=torch.zeros(graph.num_nodes, dtype=torch.long, device=generator.device)
        )
        probs = generator.classifier.predict_proba(data)
    
    # Format title
    if title is None:
        prob_str = ", ".join([f"c{i}={p:.3f}" for i, p in enumerate(probs[0].cpu().numpy())])
        title = f"Generated Graph\nPredictions: {prob_str}"
    
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.tight_layout()
        plt.show()
        return None
    
    return fig


def visualize_generation_process(
    generator: "Generator",
    num_steps: int | None = None,
    candidate_set: list[str] | None = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Visualize the complete generation process.
    
    Args:
        generator: Generator instance
        num_steps: Number of generation steps
        candidate_set: List of candidate node types
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if num_steps is None:
        num_steps = generator.max_gen_steps
    
    # Generate with intermediate steps
    graphs = generator.generate(num_steps=num_steps, return_intermediate=True)
    
    # Create subplot grid
    n_graphs = len(graphs)
    cols = min(4, n_graphs)
    rows = (n_graphs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_graphs > 1 else [axes]
    
    for i, (ax, graph) in enumerate(zip(axes, graphs)):
        visualize_generated_graph(
            generator, graph,
            candidate_set=candidate_set,
            ax=ax, show=False,
            title=f"Step {i}",
        )
    
    # Hide unused axes
    for ax in axes[n_graphs:]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_sheaf_generation_process(
    generator: "Generator",
    num_steps: int | None = None,
    candidate_set: list[str] | None = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Visualize the complete generation process.
    
    Args:
        generator: Generator instance
        num_steps: Number of generation steps
        candidate_set: List of candidate node types
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if num_steps is None:
        num_steps = generator.max_gen_steps
    
    # Generate with intermediate steps
    graphs = generator.generate(num_steps=num_steps, return_intermediate=True)
    
    probs_history = []
    reward_history = []
    for graph in graphs:
        generator.classifier.eval()
        with torch.no_grad():
            probs = generator.get_probs(graph, generator.classifier)[0].cpu().numpy()
            reward, _ = generator.calculate_reward(graph)
            reward = reward.item()
            probs_history.append(probs)
            reward_history.append(reward)

    # Create subplot grid
    n_graphs = len(graphs)
    cols = min(4, n_graphs)
    rows = (n_graphs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_graphs > 1 else [axes]
    
    for i, (ax, graph) in enumerate(zip(axes, graphs)):
        visualize_sheaf_generated_graph(
            generator, graph,
            candidate_set=candidate_set,
            ax=ax, show=False,
            title=f"Step {i}: \nP(c0)={probs_history[i][0]:.3f}, P(c1)={probs_history[i][1]:.3f}\n Reward={reward_history[i]:.3f}",
            probs=probs_history[i],
        )
    
    # Hide unused axes
    for ax in axes[n_graphs:]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":  # test bond drawing for edge types 0-3
    # Minimal dummy generator that only provides a candidate set
    class DummyGen:
        def __init__(self, candidate_set):
            self.candidate_set_raw = candidate_set

    # Lightweight containers to mimic expected graph/data attributes
    class MockData:
        def __init__(self, edge_index, edge_type, node_type):
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.node_type = node_type

    class MockSheafGraph:
        def __init__(self, num_nodes, data):
            self.num_nodes = num_nodes
            self.data = data

    # Candidate set (type strings like 'C.4' are split to get atom names)
    candidate_set = ['C.4', 'O.3', 'N.2', 'Cl.1']

    # Define node pairs with explicit bond types (0:dotted,1:single,2:double,3:triple)
    # We'll add each edge in both directions to emulate the bidirectional representation
    pairs = [
        (0, 1, 0),  # dotted
        (1, 2, 1),  # single
        (2, 3, 2),  # double
        (3, 4, 3),  # triple
        (4, 5, 0),  # dotted
        (5, 6, 2),  # double
        (6, 7, 3),  # triple
    ]

    edges = []
    edge_types = []
    for (u, v, t) in pairs:
        edges.append([u, v])
        edges.append([v, u])
        edge_types.append(t)
        edge_types.append(t)

    print(edge_types, flush=True)

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape [2, num_cols]
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    # Assign node types cycling through the candidate_set indices
    node_type = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

    data = MockData(edge_index=edge_index, edge_type=edge_type, node_type=node_type)
    graph = MockSheafGraph(num_nodes=8, data=data)
    gen = DummyGen(candidate_set)

    # Visualize
    visualize_sheaf_generated_graph(gen, graph=graph, candidate_set=candidate_set)