#!/bin/bash

# Arrays of values to sweep
d_values=(2 4 8)
layers_values=(5 10)
hidden_channels_values=(40 80)

# Path to your SLURM job script
JOB_SCRIPT="exp/scripts/run_ieee24_node.sh"

# Loop over all combinations
for d in "${d_values[@]}"; do
  for layers in "${layers_values[@]}"; do
    for hidden in "${hidden_channels_values[@]}"; do
      sbatch "$JOB_SCRIPT" --d "$d" --layers "$layers" --hidden_channels "$hidden"
      echo "Queued: d=$d, layers=$layers, hidden_channels=$hidden"
    done
  done
done