# SLURM Guide

This guide provides an overview of how to use SLURM for job scheduling and management in LUIS (Uni Hannover)

## Introduction
SLURM (Simple Linux Utility for Resource Management) is an open-source job scheduler used by many high-performance computing (HPC) clusters. It manages job submissions, resource allocation, and job monitoring.

## Basic Commands
Here are some basic SLURM commands:

- `si`: Display information about the SLURM nodes and partitions.
- `scancel`: Cancel a job.
- `sqh`: Display the list of jobs that are currently running.

## Running Interactive Session
To run an interactive session, use the `srun` command:

```bash
srun --gpus=1 --nodelist=node-name --pty /bin/bash
```

## Submitting Jobs
To submit a job, use the `sbatch` command followed by a job script:

```bash
sbatch my_job_script.sh
```

## Job Scripts
A job script is a shell script that includes SLURM directives. Here is an example:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=01:00:00
#SBATCH --gpus=1 # Request 1 GPU
#SBATCH --nodelist=node1 # Request a specific node

# Your commands go here
module load python
python my_script.py
```