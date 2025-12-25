DQN Agent for Equitable HRC Task Allocation
Author: [Your Name]
Reference: [Your Paper Title]
"""

import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    """Deep Q-Network for HRC task allocation"""
    def __init__(self, state_dim=5, action_dim=4, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ... [rest of your implementation]