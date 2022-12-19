#!/bin/bash

#SBATCH --job-name test_tf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

sh scripts/move_models.sh ./checkpoints/Transformer_2019-xx-xx_xx-xx-xx/G_BA_xx.pth ./checkpoints/Decoder_2019-xx-xx_xx-xx-xx/xx_net_G.pth trump
sh scripts/test.sh

exit 0