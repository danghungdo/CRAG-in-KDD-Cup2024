#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=ailab-rag             # A nice readable name of your job, to see it in the queue
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=8           # Number of CPUs to request
#SBATCH --gpus=1                    # Number of GPUs to request
#SBATCH --time=1-0                  # 1 day
#SBATCH --nodelist=gpunode102       # Name of the node you want to request
module load mamba

# Activate your environment, you have to create it first
micromamba activate metarag

# Your job script goes below this line
ray start --head

python main.py

ray stop

