#!/bin/bash
#! Example SLURM job script adapted for powergrid classification

#SBATCH -J uk
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

cd "$workdir"
mkdir -p logs
JOBID=$SLURM_JOB_ID

CMD="bash exp/scripts/run_uk.sh > logs/out.uk.$JOBID"

echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Command: $CMD"

eval $CMD
