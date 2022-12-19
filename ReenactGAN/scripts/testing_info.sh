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

sh scripts/move_models.sh ./checkpoints/Transformer_2019-xx-xx_xx-xx-xx/G_BA_xx.pth ./checkpoints/Decoder_2019-xx-xx_xx-xx-xx/xx_net_G.pth trump
sh scripts/test.sh

exit 0


sbatch scripts/train_Transformer.sh
sbatch scripts/train_Decoder.sh
sbatch scripts/test.sh

sh scripts/train_Transformer.sh
sh scripts/train_Decoder.sh
sh scripts/move_models.sh ./checkpoints/Transformer_2022-11-02_20-40-26/G_BA_68000.pth ./checkpoints/Decoder_2022-11-03_16-25-17/latest_net_G.pth trump
sh scripts/test.sh

srun --gres=gpu:1 --cpus-per-gpu=4 --mem-per-gpu=5G --partition debug_sw_ugrad --pty bash

--gpu_ids 0 \

--which_decoder trumpcelebv \
--which_transformer trumptmp2 &

export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"