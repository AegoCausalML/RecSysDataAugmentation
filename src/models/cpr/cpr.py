from typing import List

import torch

from utils.constants import SEED
from models.ps.ps import P_S_Network

torch.manual_seed(SEED)

def sample_alpha_posterior(n_items: int):
    return torch.randn(n_items + 1)

def sample_beta_posterior(k: int):
    return torch.randn(k)

def gen_r_from_tau(tau: torch.tensor, Q: torch.tensor, wR: torch.tensor, k: int, n_items: int):
    """
    tau: action center - torch.tensor(1, emb_dim)
    k: number of items - int
    """
    alpha = sample_alpha_posterior(n_items)

    score = (tau @ Q.T) + (wR * alpha)

    scores_dict = dict(enumerate(score[0]))
    sorted_dict = dict(sorted(scores_dict.items(), key=lambda item:item[1], reverse=True))

    return list(sorted_dict.keys())[:k]

def gen_s(ps: P_S_Network, u: int, r: List[int], M: int):
    """
    u: user id - int
    r: list of item ids - [int]
    M - itens to be selected - int
    """
    beta = sample_beta_posterior(len(r))
    
    u_b = torch.LongTensor([[u]])
    r_b = torch.LongTensor([r])
    r_mask_b = torch.ones(1, len(r))
    beta_b = beta.unsqueeze(0)

    score = ps(u_b, r_b, r_mask_b, None, beta_b).data[0]
    
    scores_dict = dict(zip(r, score))
    sorted_dict = dict(sorted(scores_dict.items(), key=lambda item:item[1], reverse=True))
    
    return list(sorted_dict.keys())[:M]