#!/bin/bash
# Build PyG CUDA extensions from source on a GPU node.
# Edit SBATCH lines for your cluster (account/partition/time/gpu).

#SBATCH --job-name=build-pyg
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -p gpu

set -euo pipefail

# Load your cluster's CUDA + compiler modules.
# Example (edit for your environment):
# module purge
# module load cuda/12.8
# module load gcc/11

cd /home/$USER/Heterogeneous-Polynomial-Sheaf-Diffusion

# Ensure the project env exists.
uv sync

# Install CUDA-enabled PyTorch (matches cu128 in this repo).
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  --force-reinstall torch==2.8.0 torchvision torchaudio

# Build PyG extensions from source (no prebuilt wheels).
export FORCE_CUDA=1
# Set this to your GPU arch if needed, e.g. "8.0" for A100.
# export TORCH_CUDA_ARCH_LIST="8.0"
uv pip install --no-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster,torch-spline-conv \
  --no-build-isolation --force-reinstall \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
