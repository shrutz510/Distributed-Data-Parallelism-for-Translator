#!/bin/bash

#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=multi_ddp
#SBATCH --output=multi_ddp

module purge

singularity exec --nv \
            --overlay /scratch/sgw6735/ddpProject/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python multi_run.py 15 5 32 1"

singularity exec --nv \
            --overlay /scratch/sgw6735/ddpProject/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python multi_run.py 15 5 64 1"


singularity exec --nv \
            --overlay /scratch/sgw6735/ddpProject/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python multi_run.py 15 5 128 1"

singularity exec --nv \
            --overlay /scratch/sgw6735/ddpProject/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python multi_run.py 15 5 32 2"

singularity exec --nv \
            --overlay /scratch/sgw6735/ddpProject/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python multi_run.py 15 5 32 4"
