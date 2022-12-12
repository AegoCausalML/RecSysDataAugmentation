import torch
from torch import nn
import torch.nn.functional as F
import math

from utils.constants import SEED

torch.manual_seed(SEED)

def normal(x, mu, sigma_sq):
    a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*math.pi).sqrt()
    return a*b

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
        
        self.linear1 = nn.Linear(n_users, hidden_dimension)
        self.linear_mu = nn.Linear(hidden_dimension, d_R)
        self.linear_sigma = nn.Linear(hidden_dimension, d_R)

    def forward(self, xb):
        xb = self.one_hot(xb, num_classes=self.n_users).to(torch.float32)
        xb = F.relu(self.linear1(xb))
        mu = self.linear_mu(xb)
        sigma_sq = self.linear_sigma(xb)
        return mu, sigma_sq
    
    def act(self, xb):
        # Batch of users
        mu, sigma_sq = self.forward(xb)
        sigma_sq = F.softplus(sigma_sq)
        eps = torch.randn(mu.size())
        action = (mu + sigma_sq.sqrt()*eps).data
        log_prob = normal(action, mu, sigma_sq).log()
        return action, log_prob