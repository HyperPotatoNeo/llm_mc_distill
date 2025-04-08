#!/bin/bash
#SBATCH --job-name=train_kl_via_entropy
#SBATCH --output=logs/mmlu_eval_%j.out
#SBATCH --error=logs/mmlu_eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=4



mkdir -p logs
module --quiet load anaconda/3
# conda activate base


# python base_mmlu.py \
#     --model_name "Qwen/Qwen2.5-3B-Instruct" \
#     --output_dir "results" \
#     --eval_split "validation" \

# python cot_mmlu.py \
#     --model_name "Qwen/Qwen2.5-3B-Instruct" \
#     --output_dir "results" \
#     --eval_split "test" \
#     --num_gpus 4 \
#     --final_forward_batch_size 2

python train_discrepancy_pred.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --json_path "results/test_mmlu_discrepancy.json" \
    --discrepancy "ce_and_entropy" \ 
    --num_epochs 25 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --accumulate_steps 2 \
    --output_dir "$SCRATCH/mc_distill" \
    --save_every 1 \
    # --debug \
    # --debug_samples 8

# python predict_discrepancy.py \
#     --discrepancy "kl" \
#     --eval_split "test" \
#     --json_path "results/" \
#     --checkpoint "$SCRATCH/mc_distill/" \
#     --model_name "Qwen/Qwen2.5-3B-Instruct" \
#     --output_path "results/"
#     --num_gpus 4

echo "Finished running"
