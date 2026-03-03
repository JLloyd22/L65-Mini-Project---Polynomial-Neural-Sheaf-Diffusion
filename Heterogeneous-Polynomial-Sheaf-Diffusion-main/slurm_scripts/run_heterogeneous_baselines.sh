#!/bin/bash
# Run all heterogeneous GNN baselines on node classification

echo "Running HAN..."
uv run python polynsd/scripts/run_gnn_nc.py model=han dataset.homogeneous=false

echo "Running HGT..."
uv run python polynsd/scripts/run_gnn_nc.py model=hgt dataset.homogeneous=false

echo "Running RGCN..."
uv run python polynsd/scripts/run_gnn_nc.py model=rgcn dataset.homogeneous=false

echo "Running HeteroGNN..."
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_gcn dataset.homogeneous=false

echo "Running HeteroGAT..."
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_gat dataset.homogeneous=false

echo "Running HeteroGIN..."
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_gin dataset.homogeneous=false

echo "Running HeteroGraphSAGE..."
uv run python polynsd/scripts/run_gnn_nc.py model=hetero_sage dataset.homogeneous=false

echo "All heterogeneous baseline runs completed!"
