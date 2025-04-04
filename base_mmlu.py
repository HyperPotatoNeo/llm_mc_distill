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

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen Instruct model on MMLU using log probabilities.")
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
        "--eval_split",
        type=str,
        default="test",
        help="test or validation set",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    return parser.parse_args()

def create_chat_prompt(question: str, choices: List[str]) -> List[Dict[str, str]]:
    """Create a chat-formatted prompt for the question."""
    choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
    
    # Create few-shot examples
    few_shot_examples = (
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
        "Answer: 2\n\n"
        "Example 3:\n"
        "Question: What is 10 - 5?\n"
        "Choices:\n"
        "0: 4\n"
        "1: 6\n"
        "2: 7\n"
        "3: 5\n\n"
        "Answer: 3\n\n"
    )
    
    user_message = (
        few_shot_examples +
        'Now answer the following question. ONLY RESPOND WITH THE OPTION INDEX NUMBER:\n\n' +
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n\n" + "Answer: "
    )
    
    # Format as a chat conversation
    return [{"role": "user", "content": user_message}]

def get_option_probabilities_batch(model, tokenizer, messages_list: List[List[Dict[str, str]]]):
    """
    Calculate the likelihood of each answer option for a batch of chat messages.
    
    Assumes each question has exactly 4 options.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        messages_list: A list where each element is the chat messages (list of dicts) for a sample.
        
    Returns:
        A list (length=batch_size) of lists containing probabilities (via softmax on logits) for each of the 4 answer options.
    """
    # Create prompt strings for all messages in the batch
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    # Tokenize the batch with padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits  # shape: (batch_size, seq_length, vocab_size)
    batch_size = logits.size(0)
    
    # Determine the index of the last token (non-padding) for each example
    lengths = inputs.attention_mask.sum(dim=1)  # shape: (batch_size)
    
    batch_option_probs = []
    for i in range(batch_size):
        last_index = lengths[i] - 1  # index of the last non-pad token
        logits_i = logits[i, last_index, :]  # logits for the last token
        
        # Compute option token ids for 4 options (0, 1, 2, 3)
        option_token_ids = [tokenizer.encode(str(j), add_special_tokens=False)[0] for j in range(4)]
        
        # Gather logits corresponding to the option token ids and compute softmax over them
        option_logits = torch.stack([logits_i[token_id] for token_id in option_token_ids])
        option_probs = torch.nn.functional.softmax(option_logits, dim=0)
        
        batch_option_probs.append(option_probs.tolist())
    
    return batch_option_probs

def evaluate_mmlu(model, tokenizer, args):
    """Evaluate the model on MMLU dataset."""
    logging.info(f"Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all")
    eval_split = args.eval_split#"test"
    
    # Filter subjects if specified
    if args.subjects:
        subject_list = args.subjects.split(",")
        logging.info(f"Evaluating on specific subjects: {subject_list}")
        filtered_dataset = ds[eval_split].filter(lambda x: x["subject"] in subject_list)
    else:
        filtered_dataset = ds[eval_split]
    
    subjects = {}
    total_correct = 0
    total_questions = 0
    
    # Group questions by subject
    for example in filtered_dataset:
        subject = example["subject"]
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append(example)
    
    # Create a dictionary to store results
    results = {
        "model_name": args.model_name,
        "overall_accuracy": 0,
        "subjects": {},
        "samples": []
    }
    
    # Evaluate each subject
    for subject, examples in tqdm(subjects.items(), desc="Evaluating subjects"):
        logging.info(f"Evaluating subject: {subject}")
        correct = 0
        
        # Process in batches
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            # Create a list of chat messages for the batch
            messages_list = [create_chat_prompt(example["question"], example["choices"]) for example in batch]
            
            # Get probabilities for each option for the entire batch in one forward pass
            batch_option_probs = get_option_probabilities_batch(model, tokenizer, messages_list)
            
            for example, option_probs in zip(batch, batch_option_probs):
                ground_truth = example["answer"]
                predicted_answer = int(np.argmax(option_probs))
                is_correct = (predicted_answer == ground_truth)
                if is_correct:
                    correct += 1
                    total_correct += 1
                total_questions += 1
                
                sample_log = {
                    "subject": subject,
                    "question": example["question"],
                    "choices": example["choices"],
                    "ground_truth": ground_truth,
                    "predicted_answer": predicted_answer,
                    "option_probs": option_probs,
                    "is_correct": is_correct
                }
                results["samples"].append(sample_log)
        
        # Calculate subject accuracy
        subject_accuracy = (correct / len(examples)) * 100
        results["subjects"][subject] = {
            "accuracy": subject_accuracy,
            "correct": correct,
            "total": len(examples)
        }
        logging.info(f"Subject {subject} accuracy: {subject_accuracy:.2f}% ({correct}/{len(examples)})")
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_questions) * 100
    results["overall_accuracy"] = overall_accuracy
    results["total_correct"] = total_correct
    results["total_questions"] = total_questions
    
    logging.info(f"Overall accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})")
    
    return results

def main():
    args = parse_args()
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean model name for saving results
    clean_model_name = args.model_name.replace("/", "-")
    
    logging.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Evaluate on MMLU
    results = evaluate_mmlu(model, tokenizer, args)
    
    # Save results
    results_path = os.path.join(args.output_dir, f"{args.eval_split}_{clean_model_name}_mmlu_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
