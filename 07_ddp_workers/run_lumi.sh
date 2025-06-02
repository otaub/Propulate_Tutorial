#!/bin/bash
#SBATCH --account=project_465001989
#SBATCH --reservation=HPO_tutorial   # comment this out if the reservation is no longer available
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:30:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
source /scratch/${SLURM_JOB_ACCOUNT}/${USER}/pvenv/bin/activate

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

set -xv


srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS bash -c " \
    RANK=\$SLURM_PROCID \
    LOCAL_RANK=\$SLURM_LOCALID \
    python search.py"
