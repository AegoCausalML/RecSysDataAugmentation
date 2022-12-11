import torch
from torch import nn
import torch.nn.functional as F

class GaussianPolicy(nn.Module):
    def __init__(self, n_users: int, d_R: int, hidden_dimension: int):
        '''
        Params:
            - n_users: número de usuários
            - d_R: dimensão da representação dos usuários
            - hidden_dimension: número de neurôneos na camada oculta
        '''
        
        super().__init__()
        
        self.n_users = n_users
        self.d_R = d_R
        self.hidden_dimension = hidden_dimension

        self.one_hot = F.one_hot

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_users, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, d_R)
        )

    def forward(self, xb):
        xb = self.one_hot(xb, num_classes=self.n_users).to(torch.float32)
        tau = self.linear_relu_stack(xb)
        return tau
