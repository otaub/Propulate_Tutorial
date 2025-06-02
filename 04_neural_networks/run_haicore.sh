#!/bin/bash
##SBATCH --reservation=haicon-gpu8   # comment this out if the reservation is no longer available
#SBATCH --partition=advanced-gpu8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=125G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:full:8

## TODO load modules
## TODO load virtual environment

srun search.py
