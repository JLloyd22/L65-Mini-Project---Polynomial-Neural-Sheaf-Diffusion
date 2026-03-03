#!/bin/bash
#! Example SLURM job script adapted for powergrid node regression

#SBATCH -J ieee39_node
#SBATCH -A MLMI-jel90-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

# TODO: update repo path if needed
workdir="$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1

cd "$workdir"
mkdir -p logs
JOBID=$SLURM_JOB_ID

CMD="bash exp/scripts/run_ieee39_node.sh > logs/out.ieee39_node.$JOBID"

echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Command: $CMD"

eval $CMD
