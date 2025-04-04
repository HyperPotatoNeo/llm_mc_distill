import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------------------------
# Data Loading
# ---------------------------
parser = argparse.ArgumentParser(description="Evaluate predicted discrepancy measure")
parser.add_argument("--discrepancy", type=str, default="kl", choices=["kl", "ce"], help="Choice of target discrepancy measure")
parser.add_argument("--eval_split", type=str, default="validation", help="test or validation set")
args = parser.parse_args()
eval_split = args.eval_split  # "test"
with open('results/' + eval_split + '_mmlu_pred_' + args.discrepancy + '.json', 'r') as f:
    mmlu_pred_kl_data = json.load(f)
with open('results/' + eval_split + '_Qwen-Qwen2.5-3B-Instruct_mmlu_cot_results.json', 'r') as f:
    cot_data = json.load(f)

mmlu_pred_kl_samples = mmlu_pred_kl_data['samples']
cot_samples = cot_data['samples']

# ---------------------------
# Part A: Threshold-based Evaluation using "kl" and "pred_kl"
# ---------------------------
# (a) Using the "kl" attribute from mmlu_pred_kl.json.
pred_kl_values = [sample[args.discrepancy] for sample in mmlu_pred_kl_samples]
max_pred_kl = max(pred_kl_values)
thresholds_pred_kl = np.linspace(max_pred_kl, 0, 100)
accuracies_pred_kl = []

for thresh in thresholds_pred_kl:
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample[args.discrepancy] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_pred_kl.append(correct / len(mmlu_pred_kl_samples))

# (b) Using the "pred_kl" attribute from mmlu_pred_kl.json.
pred_pred_kl_values = [sample['pred_' + args.discrepancy] for sample in mmlu_pred_kl_samples]
max_pred_pred_kl = max(pred_pred_kl_values)
thresholds_pred_pred_kl = np.linspace(max_pred_kl, 0, 100)
accuracies_pred_pred_kl = []

for thresh in thresholds_pred_pred_kl:
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample['pred_' + args.discrepancy] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_pred_pred_kl.append(correct / len(mmlu_pred_kl_samples))

# ---------------------------
# Part B: Quantile-based Evaluation
# ---------------------------
quantile_levels = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0

# (a) Using "kl" attribute.
accuracies_quantile_kl = []
for q in quantile_levels:
    thresh = np.quantile(pred_kl_values, q)
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample[args.discrepancy] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_quantile_kl.append(correct / len(mmlu_pred_kl_samples))

# (b) Using "pred_kl" attribute.
accuracies_quantile_pred_kl = []
for q in quantile_levels:
    thresh = np.quantile(pred_pred_kl_values, q)
    print(q, thresh)
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample['pred_' + args.discrepancy] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_quantile_pred_kl.append(correct / len(mmlu_pred_kl_samples))

# (c) Random mixing: Choose cot answer randomly with probability (1 - q)
accuracies_quantile_random = []
for q in quantile_levels:
    prob_cot = 1 - q  # probability of choosing cot answer
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if np.random.rand() < prob_cot:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_quantile_random.append(correct / len(mmlu_pred_kl_samples))

# ---------------------------
# Compute Baseline Accuracies
# ---------------------------
# Accuracy when always using the sample answer (0% CoT)
accuracy_sample = np.mean([
    1 if sample['ground_truth'] == sample['predicted_answer'] else 0
    for sample in mmlu_pred_kl_samples
])
# Accuracy when always using the CoT answer (100% CoT)
accuracy_cot = np.mean([
    1 if sample['ground_truth'] == cot_samples[i]['predicted_answer'] else 0
    for i, sample in enumerate(mmlu_pred_kl_samples)
])

# ---------------------------
# Plotting the Quantile-based Results
# ---------------------------
plt.figure(figsize=(10, 6))
# For the "kl" and "pred kl" curves, we reverse the order so that the x-axis represents the % of CoT usage.
plt.plot(quantile_levels * 100, list(reversed(accuracies_quantile_kl)),
         label='True ' + args.discrepancy, marker='o', markersize=5)
plt.plot(quantile_levels * 100, list(reversed(accuracies_quantile_pred_kl)),
         label='pred ' + args.discrepancy, marker='d', markersize=5)
plt.plot(quantile_levels * 100, list(reversed(accuracies_quantile_random)),
         label='Random mix', marker='*', markersize=5)

# Add a straight line connecting the min (0% CoT) and max (100% CoT) accuracies.
plt.plot([0, 100], [accuracy_sample, accuracy_cot], 'k--', label='Baseline line')

plt.xlabel('Compute (%CoT)')
plt.ylabel('Accuracy')
plt.title('Compute (%CoT) vs. Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.savefig('results/' + eval_split + '_compute_vs_accuracy.png')
plt.close()