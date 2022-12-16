from typing import Dict

import pandas as pd

import torch
from torch.utils.data import DataLoader

from models.pr.mind_pr_dataset import MIND_P_R_Dataset, mind_p_r_collate_fn

from utils.constants import SEED
from utils.mind_dataset import MIND_Dataset

torch.manual_seed(SEED)

def pr_train_loop(dataloader, model, loss_fn, optimizer, device):
    agg_loss = 0.0
    size = len(dataloader.dataset)
    for batch, (u, pos_r, pos_r_mask, neg_r, neg_r_mask) in enumerate(dataloader):
        u, pos_r, pos_r_mask, neg_r, neg_r_mask = u.to(device), pos_r.to(device), pos_r_mask.to(device), neg_r.to(device), neg_r_mask.to(device)
        batch_sz = u.size(0)
        pos_max_len = pos_r.size(1)
        neg_max_len = neg_r.size(1)
        
        pos_alpha = torch.randn(batch_sz, pos_max_len).to(device)
        pos_scores = model(u, pos_r, pos_r_mask, pos_alpha)
        
        neg_alpha = torch.randn(batch_sz, neg_max_len).to(device)
        neg_scores = model(u, neg_r, neg_r_mask, neg_alpha)
        
        loss = loss_fn(pos_scores, neg_scores)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        agg_loss += loss.item()
        if batch % (len(dataloader)//10) == 0:
            loss, current = loss.item() / batch_sz, batch * len(u)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return agg_loss / size

def train(train_df: pd.DataFrame, 
        user_d: Dict[str, int],
        item_d: Dict[str, int],
        max_sampling: int,
        batch_size: int = 32):
    
    train_ds = MIND_Dataset(train_df, user_d, item_d)

    pr_train_ds = MIND_P_R_Dataset(train_ds, max_sampling)
    pr_train_dl = DataLoader(pr_train_ds, batch_size=batch_size, 
                            shuffle=True, collate_fn=mind_p_r_collate_fn)
 