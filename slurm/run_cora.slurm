#!/bin/bash
#! Example SLURM job script adapted for Cora classification

#SBATCH -J cora_classification
#SBATCH -A MLMI-jel90-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

# TODO: update repo path if needed
workdir="$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "GPU check:"
nvidia-smi
/home/jel90/.conda/envs/polysheaf/bin/python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device:", torch.device("cuda:0") if torch.cuda.is_available() else "cpu")
PY

cd "$workdir"
mkdir -p logs
JOBID=$SLURM_JOB_ID

CMD="bash exp/scripts/run_cora.sh > logs/out.cora_classification.$JOBID"

echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Command: $CMD"

eval $CMD
