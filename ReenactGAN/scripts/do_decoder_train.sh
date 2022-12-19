#!/bin/bash

#SBATCH --job-name test_dec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

sh scripts/train_Decoder.sh

exit 0