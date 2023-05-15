#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=t1_run
#SBATCH --gres=gpu:4

module purge

singularity exec --nv \
            --overlay /scratch/avi2011/p1/proj_pytorch.ext3:rw \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
           /bin/bash -c "source /ext3/env.sh; python multi_run.py 15 5 1"

