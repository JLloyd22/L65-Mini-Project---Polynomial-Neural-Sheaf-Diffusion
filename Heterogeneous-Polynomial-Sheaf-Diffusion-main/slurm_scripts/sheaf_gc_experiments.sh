#!/bin/bash
#SBATCH --job-name=sheaf_gc
#SBATCH --output=logs/sheaf_gc_%j.out
#SBATCH --error=logs/sheaf_gc_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Sheaf GNN Graph Classification Experiments

# Load necessary modules (adjust for your cluster)
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Experiment 1: Diagonal Sheaf on MUTAG
echo "Running Diagonal Sheaf on MUTAG..."
python polynsd/scripts/run_sheaf_gc.py \
    experiment=sheaf_gc/sheaf_diag_mutag \
    tags=[sheaf,diagonal,MUTAG,exp1]

# Experiment 2: Polynomial Sheaf on MUTAG
echo "Running Polynomial Sheaf on MUTAG..."
python polynsd/scripts/run_poly_sheaf_gc.py \
    experiment=sheaf_gc/poly_sheaf_mutag \
    tags=[poly_sheaf,MUTAG,exp2]

# Experiment 3: Bundle Sheaf on PROTEINS
echo "Running Bundle Sheaf on PROTEINS..."
python polynsd/scripts/run_sheaf_gc.py \
    experiment=sheaf_gc/sheaf_bundle_proteins \
    tags=[sheaf,bundle,PROTEINS,exp3]

echo "All experiments completed!"
