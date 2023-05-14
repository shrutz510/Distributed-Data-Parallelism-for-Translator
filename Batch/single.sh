#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=cpu
#SBATCH --output=cpu

module purge

singularity exec \
            --overlay /scratch/sgw6735/ddpProject/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python test.py"
