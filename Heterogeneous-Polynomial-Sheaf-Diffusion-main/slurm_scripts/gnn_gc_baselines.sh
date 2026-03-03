#!/bin/bash
#SBATCH --job-name=gnn_gc_baselines
#SBATCH --output=logs/gnn_gc_%j.out
#SBATCH --error=logs/gnn_gc_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# GNN Baseline Graph Classification Experiments

# Load necessary modules
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Experiment 1: GCN on MUTAG
echo "Running GCN on MUTAG..."
python polynsd/scripts/run_gnn_gc.py \
    experiment=gnn_gc/gcn_mutag \
    tags=[GNN,GCN,MUTAG,baseline]

# Experiment 2: GCN on PROTEINS
echo "Running GCN on PROTEINS..."
python polynsd/scripts/run_gnn_gc.py \
    experiment=gnn_gc/gcn_proteins \
    tags=[GNN,GCN,PROTEINS,baseline]

# Experiment 3: GIN on MUTAG
echo "Running GIN on MUTAG..."
python polynsd/scripts/run_gnn_gc.py \
    experiment=gnn_gc/gin_mutag \
    tags=[GNN,GIN,MUTAG,baseline]

echo "All baseline experiments completed!"
