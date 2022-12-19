#!/bin/bash

#SBATCH --job-name testing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -x sw10
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

sh scripts/train_Transformer.sh
sh scripts/train_Decoder.sh

sh script/move_models.sh ./checkpoints/Transformer_2019-xx-xx_xx-xx-xx/G_BA_xx.pth ./checkpoints/Decoder_2019-xx-xx_xx-xx-xx/xx_net_G.pth trump
sh script/test.sh

exit 0

sbatch scripts/do_transformer_train.sh
sbatch scripts/do_decoder_train.sh