#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=transpose
#SBATCH --reservation=eece5640
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=transpose.%j.out

cd /scratch/$USER/eece5640/transpose/

./transpose ../input/leopard.jpg out_leopard.jpg

