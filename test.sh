#!/usr/bin/bash

#SBATCH -J ME_ViT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=28G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

which python
hostname
python mevit_train.py
exit 0