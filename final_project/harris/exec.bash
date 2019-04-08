#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=transpose
#SBATCH --reservation=eece5640
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=transpose.%j.out

#cd /scratch/$USER/eece5640/transpose/
cd /home/tu.a/eece5640/hw5/ec

./sobel input/leopard.jpg out_leopard.jpg

