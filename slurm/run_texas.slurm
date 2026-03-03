#!/bin/bash
#! Standard Texas run on GPU

export WANDB_MODE=online
export WANDB_DISABLED=false
export WANDB_PROJECT=powergrids_texas
export WANDB_ENTITY=jelloyd22-university-of-cambridge

#SBATCH -J texas
#SBATCH -A MLMI-jel90-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
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

CMD="/home/jel90/.conda/envs/polysheaf/bin/python visualizations/plot_spectral_response.py \
	--cuda 0 \
	--epochs 200 \
	--early_stopping 200 \
	--lr 0.02 \
	--maps_lr 0.005 \
	--dropout 0.7 \
	--weight_decay 0.005 \
	--sheaf_decay 0.005 \
	--hidden_channels 20 \
	--left_weights True \
	--right_weights True \
	--normalised True \
	--deg_normalised False \
	--sparse_learner True \
	--polynomial_type ChebyshevType1 \
	--seeds 0,1,2 \
	--out_dir visualizations/figures/spectral_texas \
	--csv_path visualizations/figures/spectral_texas/results.csv \
	> logs/out.texas_spectral.$JOBID"

echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Command: $CMD"

eval $CMD
