import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
  def __init__(self, input_size, num_layers, hidden_size, output_size):
    super(MLP, self).__init__()
    
    self.input_linear = nn.Linear(input_size, hidden_size)

    self.hidden = nn.ModuleList()
    for _ in range(num_layers - 1):
      self.hidden.append(nn.Linear(hidden_size, hidden_size))
    
    self.output_linear = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h = F.relu(self.input_linear(x))
    
    for layer in self.hidden:
      h = F.relu(layer(h))
    
    output = self.output_linear(h)
    
    return output
