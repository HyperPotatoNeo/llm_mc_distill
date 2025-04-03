import argparse
import json
import logging
import os
from typing import Dict, List
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import vllm
from vllm import LLM, SamplingParams

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen Instruct model on MMLU using chain-of-thought reasoning and empirical answer distribution."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Name or path of the Qwen instruct model to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of specific subjects to evaluate (default: all).",
    )
    parser.add_argument(
        "--num_cot",
        type=int,
        default=10,
        help="Number of chain-of-thought completions per question.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of available gpus for vllm generation.",
    )
    return parser.parse_args()

def create_chat_prompt(question: str, choices: List[str]) -> List[Dict[str, str]]:
    """
    Create a chat-formatted prompt for a question that instructs the model
    to provide chain-of-thought reasoning. The prompt explicitly asks the model
    NOT to output the final answer index.
    """
    choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
    few_shot_examples = (
        "Answer the following multiple choice question by reasoning step by step (chain-of-thought).\n"
        "Here is an example question:\n"
        "Question: What is 3 * 3?\n"
        "Choices:\n"
        "0: 6\n"
        "1: 9\n"
        "2: 12\n"
        "3: 18\n"
        "Final answer index: 1"
    )
    user_message = (
        few_shot_examples +
        "Now answer the following question by reasoning step by step:\n\n" +
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n\n"
    )
    return [{"role": "user", "content": user_message}]

def get_cot_empirical_distribution_batch(model, tokenizer, engine, messages_list: List[List[Dict[str, str]]], num_cot: int):
    """
    For each question prompt in messages_list, generate 'num_cot' chain-of-thought (CoT)
    completions using the vllm engine. For each generated CoT, append the string
    "\nFinal answer index: " and then compute the log probability for each answer
    index token ("0", "1", "2", "3") using the base model.
    
    The final answer forward pass is split into batches.
    
    Returns:
        batch_option_probs: A list (one per prompt) of lists of probabilities for options 0-3.
        grouped_predictions: A list (one per prompt) containing the raw predicted answer indices from each CoT.
    """
    # Create base prompt strings using the chat template.
    base_prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    # Replicate each base prompt 'num_cot' times.
    prompts = []
    question_indices = []  # maps each generated prompt back to its question index
    for i, base in enumerate(base_prompts):
        for _ in range(num_cot):
            prompts.append(base)
            question_indices.append(i)
    
    # Generate chain-of-thought completions for all prompts at once.
    max_tokens = 128  # maximum tokens for chain-of-thought reasoning
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
    completions = engine.generate(prompts, sampling_params=sampling_params)
    
    # For each completion, append the fixed string to prompt the final answer log probability computation.
    final_prompts = [completion.outputs[0].text.strip() + "\nFinal answer index: " for completion in completions]
    
    # --- Final answer index forward pass is now batched ---
    predicted_tokens = []
    option_token_ids = [tokenizer.encode(str(j), add_special_tokens=False)[0] for j in range(4)]
    final_forward_batch_size = 8  # adjust this value as needed
    for i in range(0, len(final_prompts), final_forward_batch_size):
        batch_prompts = final_prompts[i: i + final_forward_batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, seq_length, vocab_size)
        lengths = inputs.attention_mask.sum(dim=1)  # shape: (batch_size)
        for j in range(len(batch_prompts)):
            last_index = lengths[j] - 1  # index of the last non-padding token
            logits_j = logits[j, last_index, :]  # logits for the next token prediction
            option_logits = torch.stack([logits_j[token_id] for token_id in option_token_ids])
            option_probs = torch.nn.functional.softmax(option_logits, dim=0)
            predicted = int(torch.argmax(option_probs))
            predicted_tokens.append(predicted)
    
    # Group the predictions by question.
    num_questions = len(base_prompts)
    grouped_predictions = [[] for _ in range(num_questions)]
    for q_idx, pred in zip(question_indices, predicted_tokens):
        grouped_predictions[q_idx].append(pred)
    
    # Compute the empirical distribution for each question.
    batch_option_probs = []
    for preds in grouped_predictions:
        counts = [preds.count(j) for j in range(4)]
        empirical = [count / num_cot for count in counts]
        batch_option_probs.append(empirical)
    
    return batch_option_probs, grouped_predictions

def evaluate_mmlu(model, tokenizer, engine, args):
    """Evaluate the model on the MMLU dataset using full-dataset chain-of-thought generations."""
    logging.info("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all")
    eval_split = "validation"
    
    # Filter subjects if specified.
    if args.subjects:
        subject_list = args.subjects.split(",")
        logging.info(f"Evaluating on specific subjects: {subject_list}")
        filtered_dataset = ds[eval_split].filter(lambda x: x["subject"] in subject_list)
    else:
        filtered_dataset = ds[eval_split]
    
    # Flatten the dataset to a list of examples.
    all_examples = list(filtered_dataset)
    logging.info(f"Total examples to evaluate: {len(all_examples)}")
    
    # Create a list of chat messages (prompts) for all examples.
    messages_list = [create_chat_prompt(example["question"], example["choices"]) for example in all_examples]
    
    # Perform vllm generation on the full dataset at once.
    batch_option_probs, batch_cot_predictions = get_cot_empirical_distribution_batch(
        model, tokenizer, engine, messages_list, args.num_cot
    )
    
    # Aggregate results.
    results = {
        "model_name": args.model_name,
        "overall_accuracy": 0,
        "subjects": {},
        "samples": []
    }
    total_correct = 0
    total_questions = len(all_examples)
    subject_data = {}
    
    for example, option_probs, cot_preds in zip(all_examples, batch_option_probs, batch_cot_predictions):
        ground_truth = example["answer"]
        predicted_answer = int(np.argmax(option_probs))
        is_correct = (predicted_answer == ground_truth)
        if is_correct:
            total_correct += 1
        
        sample_log = {
            "subject": example["subject"],
            "question": example["question"],
            "choices": example["choices"],
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "option_probs": option_probs,
            "cot_predictions": cot_preds,
            "is_correct": is_correct
        }
        results["samples"].append(sample_log)
        
        # Group results by subject.
        subject = example["subject"]
        if subject not in subject_data:
            subject_data[subject] = []
        subject_data[subject].append(sample_log)
    
    # Compute per-subject accuracy.
    for subject, samples in subject_data.items():
        correct = sum(sample["is_correct"] for sample in samples)
        total = len(samples)
        subject_accuracy = (correct / total) * 100
        results["subjects"][subject] = {
            "accuracy": subject_accuracy,
            "correct": correct,
            "total": total
        }
        logging.info(f"Subject {subject} accuracy: {subject_accuracy:.2f}% ({correct}/{total})")
    
    overall_accuracy = (total_correct / total_questions) * 100
    results["overall_accuracy"] = overall_accuracy
    results["total_correct"] = total_correct
    results["total_questions"] = total_questions
    logging.info(f"Overall accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    
    return results

def main():
    args = parse_args()
    setup_logging()
    
    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean model name for saving results.
    clean_model_name = args.model_name.replace("/", "-")
    
    logging.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Set up the vllm LLM for full-dataset chain-of-thought generation.
    logging.info("Setting up vllm LLM for chain-of-thought generation...")
    engine = LLM(args.model_name, max_num_seqs=1024, tensor_parallel_size=args.num_gpus)
    
    # Evaluate the model on MMLU.
    results = evaluate_mmlu(model, tokenizer, engine, args)
    
    # Save results.
    results_path = os.path.join(args.output_dir, f"{clean_model_name}_mmlu_cot_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
