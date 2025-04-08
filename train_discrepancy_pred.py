import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from models.regress_model import RegressionModel
from models.two_headed_regress_model import RegressionModel as TwoHeadedRegressionModel
import wandb
from dotenv import load_dotenv
load_dotenv()

# Define a Dataset that prepares the prompt and target for each sample
class MMLUKLDataset(Dataset):
    def __init__(self, json_path, tokenizer, mean, std, incontext_example, discrepancy="kl"):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.samples = data["samples"]
        self.tokenizer = tokenizer
        self.mean = mean
        self.std = std
        self.incontext_example = incontext_example
        self.discrepancy = discrepancy

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        choices = sample["choices"]
        # Format the choices as one per line with index prefixes
        choices_str = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
        prompt = (
            f"{self.incontext_example}\n\n"
            "Now answer the following question. ONLY RESPOND WITH THE OPTION INDEX NUMBER:\n\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_str}\n"
            f"Answer: "
        )
        # If the tokenizer expects a list of messages, adjust accordingly.
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        
        # Handle different discrepancy types
        if self.discrepancy == "ce_and_entropy":
            # For dual-headed model, return both cross-entropy and entropy
            ce = sample["ce"]
            entropy = sample["entropy"]
            # Normalize both targets
            ce_target = (ce - self.mean[0]) / self.std[0]
            entropy_target = (entropy - self.mean[1]) / self.std[1]
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "target_ce": torch.tensor(ce_target, dtype=torch.bfloat16),
                "target_entropy": torch.tensor(entropy_target, dtype=torch.bfloat16)
            }
        else:
            # For single-headed model, keep existing functionality
            metric = sample[self.discrepancy]
            target = (metric - self.mean) / self.std
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "target": torch.tensor(target, dtype=torch.bfloat16)
            }
    
def compute_target_stats(json_path, discrepancy="kl"):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if discrepancy == "ce_and_entropy":
        # For dual-headed model, compute statistics for both targets
        ce_metrics = [sample["ce"] for sample in data["samples"]]
        entropy_metrics = [sample["entropy"] for sample in data["samples"]]
        
        ce_mean = sum(ce_metrics) / len(ce_metrics)
        ce_std = (sum((x - ce_mean) ** 2 for x in ce_metrics) / len(ce_metrics)) ** 0.5
        
        entropy_mean = sum(entropy_metrics) / len(entropy_metrics)
        entropy_std = (sum((x - entropy_mean) ** 2 for x in entropy_metrics) / len(entropy_metrics)) ** 0.5
        
        return [ce_mean, entropy_mean], [ce_std, entropy_std]
    else:
        # For single-headed model, keep existing functionality
        metrics = [sample[discrepancy] for sample in data["samples"]]
        mean = sum(metrics) / len(metrics)
        std = (sum((x - mean) ** 2 for x in metrics) / len(metrics)) ** 0.5
        return mean, std

def save_model(unwrapped_model, args, epoch=None):
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = f"qwen_{'kl_via_entropy' if args.discrepancy == 'ce_and_entropy' else args.discrepancy}_regression"
    filename = f"{base_name}_epoch{epoch}.pt" if epoch else f"{base_name}.pt"
    output_path = os.path.join(args.output_dir, filename)
    
    torch.save(unwrapped_model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def main(args):
    # Initialize Accelerator with gradient accumulation steps
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulate_steps)
    device = accelerator.device

    # Compute the mean and std of the KL targets from the JSON file
    mean, std = compute_target_stats(args.json_path, args.discrepancy)
    #print(f"KL Mean: {mean:.4f}, KL Std: {std:.4f}")


    
    # Initialize Weights & Biases
    wandb.init(
        project="llm_mc_distill",
        name=f"run_{args.discrepancy}_regress_bs{args.batch_size}_acc{args.accumulate_steps}_lr{args.learning_rate}",
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "accumulate_steps": args.accumulate_steps,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "json_path": args.json_path,
            "output_dir": args.output_dir,
            "metric_mean": mean,
            "metric_std": std,
            "discrepancy": args.discrepancy
        }
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define the in-context example with the desired format
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

    # Create Dataset and DataLoader
    dataset = MMLUKLDataset(args.json_path, tokenizer, mean, std, incontext_example, args.discrepancy)
    
    if args.debug:
        dataset.samples = dataset.samples[:args.debug_samples]
        print(f"DEBUG MODE: Using only {len(dataset.samples)} samples")
    
    collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # Load the appropriate regression model based on discrepancy type
    if args.discrepancy == "ce_and_entropy":
        model = TwoHeadedRegressionModel(args.model_name)
    else:
        model = RegressionModel(args.model_name)
    model.to(device)

    # Set the mean/std once after model initialization
    if args.discrepancy == "ce_and_entropy":
        model.regressor_1_mean = torch.tensor(mean[0], device=device)
        model.regressor_2_mean = torch.tensor(mean[1], device=device)
        model.regressor_1_std = torch.tensor(std[0], device=device)
        model.regressor_2_std = torch.tensor(std[1], device=device)
    else:
        model.regressor_mean = torch.tensor(mean, device=device)
        model.regressor_std = torch.tensor(std, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Calculate total training steps and define warmup steps (using 10% of total steps)
    num_update_steps_per_epoch = len(dataloader) // args.accumulate_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    warmup_steps = int(0.1 * max_train_steps)
    
    # Initialize cosine LR scheduler with warmup that decays to 0 at the end of training
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=max_train_steps
    )

    criterion = nn.MSELoss()

    # Prepare model, optimizer, and dataloader for multi-GPU and mixed precision training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    print('STARTING TRAINING')

    global_step = 0
    for epoch in range(args.num_epochs):
        print('EPOCH: ', epoch)
        running_loss = 0.0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                if args.discrepancy == "ce_and_entropy":
                    # For dual-headed model
                    target_ce = batch["target_ce"].to(device).to(torch.bfloat16)
                    target_entropy = batch["target_entropy"].to(device).to(torch.bfloat16)
                    
                    # Get predictions from both heads
                    pred_ce, pred_entropy = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Calculate loss for each head
                    loss_ce = criterion(pred_ce, target_ce)
                    loss_entropy = criterion(pred_entropy, target_entropy)
                    
                    # Combined loss (equal weighting)
                    loss = loss_ce + loss_entropy
                else:
                    # For single-headed model, use existing functionality
                    targets = batch["target"].to(device).to(torch.bfloat16)
                    predictions = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(predictions, targets)
                
                accelerator.backward(loss)
                optimizer.step()
                # Only step the scheduler if gradients are synced (i.e. at an actual update step)
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad()

                running_loss += loss.item()

            # Optionally log the average loss every accumulation cycle
            if (step + 1) % args.accumulate_steps == 0:
                avg_loss = running_loss / args.accumulate_steps
                
                # Log additional metrics for the dual-headed model
                if args.discrepancy == "ce_and_entropy":
                    wandb.log({
                        "loss": avg_loss,
                        "loss_ce": loss_ce.item(),
                        "loss_entropy": loss_entropy.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "step": global_step
                    })
                else:
                    wandb.log({
                        "loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "step": global_step
                    })
                
                running_loss = 0.0

        # Checkpoint saving
        if (epoch + 1) % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_model(unwrapped_model, args, epoch + 1)

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_model(unwrapped_model, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5 3B-Instruct with a linear regression head to predict normalized discrepancy values using Accelerate for multi-GPU training with gradient accumulation."
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Pretrained model name or path")
    parser.add_argument("--json_path", type=str, default="results/test_mmlu_discrepancy.json", help="Path to the mmlu_discrepancy.json file")
    parser.add_argument("--discrepancy", type=str, default="kl", choices=["kl", "ce", "ce_and_entropy"], help="Choice of target discrepancy measure")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accumulate_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="/pscratch/sd/s/siddart2/mc_distill/kl_regression", help="Directory to save the trained model")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited samples")
    parser.add_argument("--debug_samples", type=int, default=8, help="Number of samples to use in debug mode")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
  

    args = parser.parse_args()
    main(args)
