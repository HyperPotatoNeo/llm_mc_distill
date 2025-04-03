import json
import math
import numpy as np

# Replace these file names with the actual names of your JSON files.
first_file = "results/Qwen-Qwen2.5-3B-Instruct_mmlu_results.json"  # JSON with evaluations without chain-of-thought
second_file = "results/Qwen-Qwen2.5-3B-Instruct_mmlu_cot_results.json"      # JSON with chain-of-thought evaluations

# Load the JSON data from the two files
with open(first_file, "r") as f1, open(second_file, "r") as f2:
    eval_without_cot = json.load(f1)
    eval_with_cot = json.load(f2)

# Compute forward KL for each question and add it to the first json's object.
# Forward KL: KL(p || q) = sum_i p_i * log(p_i / q_i)
for i, entry in enumerate(eval_without_cot['samples']):
    p = eval_with_cot['samples'][i]["option_probs"]  # "ground truth" distribution (chain-of-thought)
    q = entry["option_probs"]             # distribution from the first evaluation

    kl_divergence = 0.0
    for p_val, q_val in zip(p, q):
        # if p_val is zero, the term is 0 (by convention 0 * log(0/q) = 0)
        if p_val > 0:
            kl_divergence += p_val * math.log(p_val / q_val)
    
    # Add the computed kl divergence to the current entry
    entry["kl"] = kl_divergence

# Save the updated list of objects to "mmlu_kl.json"
with open("results/mmlu_kl.json", "w") as out_file:
    json.dump(eval_without_cot, out_file, indent=4)

print("Saved forward KL values to results/mmlu_kl.json")
