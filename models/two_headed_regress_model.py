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
        self.regressor_1 = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)
        self.regressor_2 = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)
        self.regressor_1_mean = None
        self.regressor_1_std = None
        self.regressor_2_mean = None
        self.regressor_2_std = None

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        features = last_hidden[:, -1, :]  # Use the last token's hidden state
        pred_1 = self.regressor_1(features).squeeze(-1) # by convention we will make this one the (normalized) cross entropy H(p,q)
        pred_2 = self.regressor_2(features).squeeze(-1) # by convention we will make this one the (normalized) entropy H(p)
        return pred_1, pred_2
    
    def predict_difference_denormalized(self, input_ids, attention_mask):
        pred_1, pred_2 = self.forward(input_ids, attention_mask)
        pred_1 = pred_1 * self.regressor_1_std + self.regressor_1_mean
        pred_2 = pred_2 * self.regressor_2_std + self.regressor_2_mean
        return pred_1 - pred_2 # this is therefore the kl divergence D(p||q)
    
