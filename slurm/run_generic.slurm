#!/bin/bash
#! Generic SLURM job script for PolySheafNeuralNetworks-PowerGrids

#SBATCH -J polysheaf
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

workdir="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

cd "$workdir"
mkdir -p logs
JOBID=$SLURM_JOB_ID

# Provide arguments via sbatch --export=ARGS="..."
if [ -z "$ARGS" ]; then
  echo "ERROR: ARGS is not set."
  echo "Example: sbatch --export=ARGS=\"--dataset=ieee118_node --task=regression --model=DiagSheafPolynomial --epochs=100 --cuda=0\" slurm/run_generic.slurm"
  exit 1
fi

CMD="python -m exp.run $ARGS > logs/out.generic.$JOBID"

echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Command: $CMD"

eval $CMD
