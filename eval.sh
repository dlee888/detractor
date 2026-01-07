#!/bin/bash
#
#SBATCH --job-name=detractor
#SBATCH --output=eval_%j.txt
#SBATCH --error=/tmp/err_%j.txt
#
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH -N 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --time 2-00:00:00

source /home/dlee888/.bashrc
cd /home/dlee888/detractor/

conda activate detractor

export PYTHONPATH=.
python -u rl/evaluate.py "$@"
