#!/bin/bash
##SBATCH --reservation=haicon   # comment this out if the reservation is no longer available
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00

## TODO load modules
## TODO load virtual environment

srun search.py
