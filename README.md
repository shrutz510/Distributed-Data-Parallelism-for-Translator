# Distributed-Data-Parallelism-for-Translator

Overview

This project implements distributed data parallelism to train a German to English translation model. The  model is implemented on different hardware configurations, including single-node multiple CPU, single-node single GPU, and multiple-node multiple GPUs. The project focuses on building a scalable deep learning pipeline using PyTorch that can handle large datasets and complex models. To determine the model’s performance we benchmarked it to determine which version produces the best performance with maximum efficiency.

Implementation

In PyTorch, Distributed Data Parallelism (DDP) is implemented using the torch.nn.parallel.DistributedDataParallel module. The program is set up on the HPC environment with the required packages in the Singularity container. The code is run for different batch sizes and number of GPUs and is parallelized to run across multiple process. The SBATCH files are created to run the different combinations of configurations. We use multiple German statements to evaluate the models. 

Code Structure

