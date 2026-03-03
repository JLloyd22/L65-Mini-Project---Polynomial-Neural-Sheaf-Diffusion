#!/bin/bash
#! Short SLURM smoke test for Texas

#SBATCH -J texas_short
#SBATCH -A MLMI-jel90-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

workdir="$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$workdir"
mkdir -p logs
JOBID=$SLURM_JOB_ID

CMD="bash exp/scripts/run_texas_short.sh > logs/out.texas_short.$JOBID"

echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Command: $CMD"

eval $CMD
