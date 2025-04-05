#!/bin/bash
#SBATCH --job-name=mmlu_eval
#SBATCH --output=logs/mmlu_eval_%j.out
#SBATCH --error=logs/mmlu_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:h100:4|gpu:a100:4|gpu:a100l:4|gpu:l40s:4

mkdir -p logs
module --quiet load anaconda/3
# conda activate base

python base_mmlu.py \
    --output_dir "path/to/output" \
    --eval_split "test" \

echo "Finished running""