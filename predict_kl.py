import json
import torch
import argparse
from transformers import AutoTokenizer
from models.regress_model import RegressionModel

def compute_target_stats(json_path):
    """Compute the mean and standard deviation of the kl values from the JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    kls = [sample["kl"] for sample in data["samples"]]
    mean = sum(kls) / len(kls)
    std = (sum((x - mean) ** 2 for x in kls) / len(kls)) ** 0.5
    return mean, std

def main():
    parser = argparse.ArgumentParser(description="Predict KL values using the trained Qwen KL predictor.")
    parser.add_argument("--json_path", type=str, default="results/mmlu_kl.json", help="Path to the mmlu_kl.json file")
    parser.add_argument("--checkpoint", type=str, default="/pscratch/sd/s/siddart2/mc_distill/kl_regression/qwen_kl_regression.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Pretrained model name or path")
    parser.add_argument("--output_path", type=str, default="results/mmlu_pred_kl.json", help="Path to save the output JSON file")
    args = parser.parse_args()

    # Compute the mean and std for de-normalizing predictions.
    mean, std = compute_target_stats(args.json_path)

    # Load the JSON data.
    with open(args.json_path, "r") as f:
        data = json.load(f)
    samples = data["samples"]

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define the same in-context example used during training.
    incontext_example = (
        "Answer the following multiple choice question by responding only with the index of the correct option. Below are some few shot examples for the format to follow.\n\n"
        "Example 1:\n"
        "Question: What is 3 * 3?\n"
        "Choices:\n"
        "0: 6\n"
        "1: 9\n"
        "2: 12\n"
        "3: 18\n"
        "4: 15\n\n"
        "Answer: 1\n\n"
        "Example 2:\n"
        "Question: What is 7 + 5?\n"
        "Choices:\n"
        "0: 10\n"
        "1: 13\n"
        "2: 12\n"
        "3: 14\n\n"
        "Answer: 2"
    )

    # Load the trained model.
    model = RegressionModel(args.model_name)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loop over each sample and predict KL.
    for sample in samples:
        question = sample["question"]
        choices = sample["choices"]
        # Format choices with index prefixes.
        choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
        # Construct the prompt in the same way as during training.
        prompt = (
            f"{incontext_example}\n\n"
            "Now answer the following question. ONLY RESPOND WITH THE OPTION INDEX NUMBER:\n\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_str}\n"
            f"Answer: "
        )
        # If the tokenizer supports a chat template, apply it.
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        # Tokenize the prompt.
        encoded = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Get the model prediction (normalized).
        with torch.no_grad():
            pred_norm = model(input_ids=input_ids, attention_mask=attention_mask)
            # Since we are processing one sample at a time, extract the scalar value.
            pred_norm = pred_norm.item()
        # Denormalize the prediction.
        pred_kl = pred_norm * std + mean

        # Add the predicted kl value to the sample.
        sample["pred_kl"] = pred_kl

    # Save the updated JSON with predictions.
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()
