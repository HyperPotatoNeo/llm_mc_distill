import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoConfig

# Define a model that wraps Qwen and adds a linear regression head
class RegressionModel(nn.Module):
    def __init__(self, model_name):
        super(RegressionModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.qwen = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        hidden_size = config.hidden_size
        # Optionally, freeze the Qwen model parameters:
        # for param in self.qwen.parameters():
        #     param.requires_grad = False
        self.regressor = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)
        self.register_buffer("regressor_mean", torch.tensor(0.0))
        self.register_buffer("regressor_std", torch.tensor(1.0))

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        features = last_hidden[:, -1, :]  # Use the last token's hidden state
        pred = self.regressor(features).squeeze(-1)
        return pred
    
    def predict_denormalized(self, input_ids, attention_mask):
        pred = self.forward(input_ids, attention_mask)
        pred = pred * self.regressor_std + self.regressor_mean
        return pred