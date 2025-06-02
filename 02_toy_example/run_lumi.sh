#!/bin/bash
#SBATCH --account=project_465001989
#SBATCH --reservation=HPO_tutorial   # comment this out if the reservation is no longer available
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
source /scratch/${SLURM_JOB_ACCOUNT}/${USER}/pvenv/bin/activate

srun python search.py
mv /tmp/pcheckpoints .
