#!/bin/bash
#SBATCH --job-name=run_cora_classification
#SBATCH --output=run_cora_%j.out
#SBATCH --error=run_cora_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load any required modules (uncomment and edit as needed)
# module load python/3.8

# Activate your environment if needed
# source ~/your_env/bin/activate

# Navigate to the script directory
cd ~/PolySheafNeuralNetworks-PowerGrids/exp/scripts

# Run the classification script
bash run_cora.sh
