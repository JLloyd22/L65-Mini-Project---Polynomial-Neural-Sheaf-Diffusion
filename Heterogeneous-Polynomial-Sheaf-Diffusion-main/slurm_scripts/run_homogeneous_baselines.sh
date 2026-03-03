#!/bin/bash
# Run all homogeneous GNN baselines on node classification

echo "Running GCN..."
uv run python polynsd/scripts/run_gnn_nc.py model=gcn dataset.homogeneous=true

echo "Running GAT..."
uv run python polynsd/scripts/run_gnn_nc.py model=gat dataset.homogeneous=true

echo "Running GIN..."
uv run python polynsd/scripts/run_gnn_nc.py model=gin dataset.homogeneous=true

echo "Running GraphSAGE..."
uv run python polynsd/scripts/run_gnn_nc.py model=sage dataset.homogeneous=true

echo "All homogeneous baseline runs completed!"
