#!/bin/bash
#SBATCH --array=1-30
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=2
#SBATCH --qos=nopreemption

base_cmd=$1 #e.g. python3 lstm.py --dataset crystal --c 
id=$SLURM_ARRAY_TASK_ID

eval "$base_cmd --ids $id"