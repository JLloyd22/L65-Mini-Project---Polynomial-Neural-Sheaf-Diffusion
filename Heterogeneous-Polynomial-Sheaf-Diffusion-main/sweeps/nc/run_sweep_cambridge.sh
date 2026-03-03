#!/bin/bash
#SBATCH -J PolySheaf_Sweep
#SBATCH --account=LIO-CHARM-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/ab3352/Heterogeneous-Polynomial-Sheaf-Diffusion/logs/%x_%j.out
#SBATCH --error=/home/ab3352/Heterogeneous-Polynomial-Sheaf-Diffusion/logs/%x_%j.err

dataset=dblp
model=diag_poly_sheaf

echo "=== Job Start: $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

REPO_DIR=/home/ab3352/Heterogeneous-Polynomial-Sheaf-Diffusion
mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR" || exit 1

# ---- Modules (edit for your cluster) ----
# module purge
# module load cuda/12.8
# module load gcc/11

echo "Python (uv): $(uv run which python)"
echo "Wandb (uv):  $(uv run which wandb || echo NOT_FOUND)"
uv run wandb --version || { echo "ERROR: wandb not available in this env"; exit 1; }

uv sync

# Build PyG CUDA extensions from source (requires CUDA toolchain on the node).
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  --force-reinstall torch==2.8.0 torchvision torchaudio
export FORCE_CUDA=1
# export TORCH_CUDA_ARCH_LIST="8.0"  # Example: A100
uv pip install --no-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster,torch-spline-conv \
  --no-build-isolation --force-reinstall \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

uv run wandb login
echo "Launching W&B Sweep/Agent..."

# --- Sweep settings (override with env vars at sbatch time) ---
WANDB_PROJECT="${WANDB_PROJECT:-Heterogeneous_Polynomial_SD}"
SWEEP_CONFIG="${SWEEP_CONFIG:-sweeps/nc/${dataset}/${model}.yaml}"
WANDB_ENTITY="${WANDB_ENTITY:-sheaf_hypergraphs}"
SWEEP_ID="${SWEEP_ID:-}"

# Create a sweep if SWEEP_ID is not provided.
if [ -z "$SWEEP_ID" ]; then
  echo "Creating sweep from ${SWEEP_CONFIG}..."
  SWEEP_ID="$(uv run wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG" | awk '/Creating sweep with ID:/ {print $NF}')"
fi

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Failed to determine SWEEP_ID."
  exit 1
fi

echo "Running agent for: ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
uv run wandb agent "${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"

echo "=== Job End: $(date) ==="
