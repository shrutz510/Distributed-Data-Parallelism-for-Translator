## Distributed-Data-Parallelism-for-Translator


## Overview

This project implements distributed data parallelism to train a German to English translation model. The  model is implemented on different hardware configurations, including single-node multiple CPU, single-node single GPU, and multiple-node multiple GPUs. The project focuses on building a scalable deep learning pipeline using PyTorch that can handle large datasets and complex models. To determine the model’s performance we benchmarked it to determine which version produces the best performance with maximum efficiency.

## Implementation

In PyTorch, Distributed Data Parallelism (DDP) is implemented using the torch.nn.parallel.DistributedDataParallel module. The program is set up on the HPC environment with the required packages in the Singularity container. The code is run for different batch sizes and number of GPUs and is parallelized to run across multiple process. The SBATCH files are created to run the different combinations of configurations. We use multiple German statements to evaluate the models. 


## Code Structure

```
├── ...
├── src # Source files for the program
│ ├── help.py # contains functions returning the model and dataloader to each spawn
│ ├── multi_run.py # contains function that is spawned in each process, calls the trainer class
│ └── translate.py # load model and interact with the translator
│ └── cpu_run.py # non DDP training on CPU
| └── model.py # create the transformer model
│ └── outputs # contains checkpoints saved during training
│   ├── ...
│   ├── ...
│   └── ...
│
└── Batch # SBATCH files required to run training across specified resources
│   ├── 1GPU.SBATCH
│   ├── 2GPU.SBATCH
│   └── 4GPU.SBATCH
│   └── CPU.SBATCH    
│
└── Evaluation
│   ├── GoogleTranslate.mov    #Google Translate vs translator evaluation for certain inputs
│
└── Logs
│   ├── ...
│   ├── ...
│   └── ...  

```

## Insturctions to Run Training
 - Make desired changes to the model.py file
 - Set the desired parameters for the model in help.py file
 - Copy the structure to you HPC cluster
 - Change the .SBATCH file to reflect you Singluarity Instance
 - Submit the jobs
 
## Insturctions to Run Inferece
- In the translate.py file, load the model that you want
- Run the file
- Input q to end the interactive translator

