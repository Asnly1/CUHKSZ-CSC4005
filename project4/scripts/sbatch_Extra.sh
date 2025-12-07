#!/bin/bash
#SBATCH -o ./Project4-results-Extra.txt
#SBATCH -p Release
#SBATCH -J Project4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Necessary Environment Variables for Triton


echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "------------------------------------------------------------------"

echo ">>> Running Extra: Triton Flash Attention V2"

srun -n 1 --gpus 1 python3 ./extra/flash_attention_v2.py

echo "------------------------------------------------------------------"

echo ""
echo "Job finished at: $(date)"
