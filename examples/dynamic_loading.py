import torch
import torch.nn as nn
import os
import json

from flashtensors.torch_storage import save_dict, load_dict

class SimpleModel(nn.Module):
    def __init__(self, size=(3,3)):
        super(SimpleModel, self).__init__()
        # Create a single parameter tensor of shape (3, 3)
        self.weight = nn.Parameter(torch.randn(*size))
        
    def forward(self, x):
        return x @ self.weight  # Simple matrix multiplication

model = SimpleModel()

state_dict = model.state_dict()

save_dict(state_dict, "/workspace/test_model")

new_state_dict = load_dict("/workspace/test_model", {"":0}, "/workspace")

device = torch.device("cuda")
model.to()
print(state_dict)

print(new_state_dict)

model.load_state_dict(new_state_dict)

input("Continue")

x= torch.tensor([1.0,1.0,1.0])

response = model(x)

print(response)

input("Continue")

