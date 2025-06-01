#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=0:05:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
source /scratch/${SLURM_JOB_ACCOUNT}/${USER}/pvenv/bin/activate

srun python search.py
mv /tmp/pcheckpoints .
