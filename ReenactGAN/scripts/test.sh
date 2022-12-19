#!/bin/bash

#SBATCH --job-name test_code
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

python test.py \
--model_dir ./pretrained_models/ \
--root_dir ./test_imgs/samples/image/ \
--name_list ./test_imgs/samples/images_list.txt \
--save_root_path results-celebv-tmp2/ \
--real_F1_path real_F1 \
--Boundary_path Boundary \
--Boundary_transformed_path Boundary_transformed \
--fake_F2_path fake_F2 \
--gpu_ids -1 \
--dataset_mode single \
--norm batch \
--batchSize 1 \
--nThreads 1 \
--fineSize_F1 256 \
--fineSize_Boundary 64 \
--nc_F1 3 \
--nc_F2 3 \
--nc_Boundary 15 \
--num_stacks 2 \
--num_blocks 1 \
--which_boundary_detection v8 \

wait