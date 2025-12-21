#!/bin/bash
#
#SBATCH --job-name=detractor
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#
#SBATCH --partition=mit_preemptable
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100GB
#SBATCH --time 1-00:00:00

source /home/dlee888/.bashrc
cd /home/dlee888/detractor/

conda activate detractor

export PYTHONPATH=.
python -u rl/train.py
