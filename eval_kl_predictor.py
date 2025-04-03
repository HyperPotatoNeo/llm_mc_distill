import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Data Loading
# ---------------------------
# Load the JSON files. All files should have samples in the same order.
with open('results/mmlu_pred_kl.json', 'r') as f:
    mmlu_pred_kl_data = json.load(f)
with open('results/Qwen-Qwen2.5-3B-Instruct_mmlu_cot_results.json', 'r') as f:
    cot_data = json.load(f)

mmlu_pred_kl_samples = mmlu_pred_kl_data['samples']
cot_samples = cot_data['samples']

# ---------------------------
# Part A: Threshold-based Evaluation using "kl" and "pred_kl"
# ---------------------------
# (a) Using the "kl" attribute from mmlu_pred_kl.json.
pred_kl_values = [sample['kl'] for sample in mmlu_pred_kl_samples]
max_pred_kl = max(pred_kl_values)
thresholds_pred_kl = np.linspace(0, max_pred_kl, 100)
accuracies_pred_kl = []

for thresh in thresholds_pred_kl:
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample['kl'] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_pred_kl.append(correct / len(mmlu_pred_kl_samples))

# (b) Using the "pred_kl" attribute from mmlu_pred_kl.json.
pred_pred_kl_values = [sample['pred_kl'] for sample in mmlu_pred_kl_samples]
max_pred_pred_kl = max(pred_pred_kl_values)
thresholds_pred_pred_kl = np.linspace(0, max_pred_pred_kl, 100)
accuracies_pred_pred_kl = []

for thresh in thresholds_pred_pred_kl:
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample['pred_kl'] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_pred_pred_kl.append(correct / len(mmlu_pred_kl_samples))

# ---------------------------
# Plotting the Threshold-based Results
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(thresholds_pred_kl, accuracies_pred_kl, label='True kl', marker='x', markersize=3)
plt.plot(thresholds_pred_pred_kl, accuracies_pred_pred_kl, label='pred kl', marker='s', markersize=3)
plt.xlabel('KL Threshold')
plt.ylabel('Accuracy')
plt.title('KL Threshold vs. Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.savefig('results/true_vs_pred_kl.png')
plt.close()

# ---------------------------
# Part B: Quantile-based Evaluation
# ---------------------------
# For each quantile (0%, 10%, ..., 100%) of the "kl" values, we will:
#   - Compute the quantile threshold.
#   - Use the cot answer for samples with value > threshold,
#     and the sample answer otherwise.
# Then compute accuracy.

quantile_levels = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0

# (a) Using "kl" attribute.
accuracies_quantile_kl = []
for q in quantile_levels:
    thresh = np.quantile(pred_kl_values, q)
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample['kl'] > thresh:
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
    correct = 0
    for i, sample in enumerate(mmlu_pred_kl_samples):
        if sample['pred_kl'] > thresh:
            pred = cot_samples[i]['predicted_answer']
        else:
            pred = sample['predicted_answer']
        if sample['ground_truth'] == pred:
            correct += 1
    accuracies_quantile_pred_kl.append(correct / len(mmlu_pred_kl_samples))

# ---------------------------
# Plotting the Quantile-based Results
# ---------------------------
plt.figure(figsize=(10, 6))
# Multiply quantile levels by 100 for percentage display.
plt.plot(quantile_levels * 100, list(reversed(accuracies_quantile_kl)), label='True kl', marker='o', markersize=5)
plt.plot(quantile_levels * 100, list(reversed(accuracies_quantile_pred_kl)), label='pred kl', marker='d', markersize=5)
plt.xlabel('Compute (%CoT)')
plt.ylabel('Accuracy')
plt.title('Quantile vs. Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.savefig('results/compute_vs_accuracy.png')
plt.close()
