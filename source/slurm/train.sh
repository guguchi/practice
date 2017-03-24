#!/bin/bash
#SBATCH --nodelist=kng03
#SBATCH --gres=gpu:1
#SBATCH --mem 15000
#SBATCH -c 2
#SBATCH -t 4800

python ten.py