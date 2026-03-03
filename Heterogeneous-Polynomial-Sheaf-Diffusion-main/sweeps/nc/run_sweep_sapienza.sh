#!/bin/bash
#SBATCH --job-name=heterogeneous_polynomial_neural_sheaf_diffusion_ablation
#SBATCH --partition=queue_dip_ingegneria
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/proj/diaghpc/aborgi/Heterogeneous-Polynomial-Sheaf-Diffusion/logs/%x_%j.out
#SBATCH --error=/mnt/beegfs/proj/diaghpc/aborgi/Heterogeneous-Polynomial-Sheaf-Diffusion/logs/%x_%j.err
#SBATCH --account=diaghpc

dataset=dblp
model=diag_poly_sheaf

# Make sure logs directory exists
mkdir -p logs

# Load your environment (miniconda + myenv via .bashrc)
#source ~/.bashrc
CONDA_ROOT_PATH="/cm/shared/apps/linux-ubuntu22.04-zen2/miniconda3/24.3.0/4ycfox6czb6abkkr2x2bvs6gsmwnlqnn"

# 2. EXPLICITLY initialize Conda for the current non-interactive shell
source "$CONDA_ROOT_PATH/etc/profile.d/conda.sh"

# 3. ACTIVATE the environment where 'wandb' is installed (likely 'base')
conda activate esnn

echo "===== ENV CHECK ====="
echo "Host: $(hostname)"
echo "Python: $(which python)"
python - << 'EOF'
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__file__)
EOF
echo "====================="
echo "=== Job Start: $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

REPO_DIR=/mnt/beegfs/proj/diaghpc/aborgi/Heterogeneous-Polynomial-Sheaf-Diffusion
mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR" || exit 1

# ---- Modules (edit for your cluster) ----
# module purge
# module load cuda/12.8
# module load gcc/11

uv sync

echo "Python (uv): $(uv run which python)"
echo "Wandb (uv):  $(uv run which wandb || echo NOT_FOUND)"
uv run wandb --version || { echo "ERROR: wandb not available in this env"; exit 1; }

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
  SWEEP_ID="$(
  uv run wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1 \
  | awk -F': ' '/Creating sweep with ID:/ {print $NF}' \
  | tail -n1
  )"
fi


if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Failed to determine SWEEP_ID."
  exit 1
fi

echo "Running agent for: ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
uv run wandb agent "${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"

echo "=== Job End: $(date) ==="
