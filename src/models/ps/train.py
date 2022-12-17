import torch
from torch.utils.data import DataLoader

import pandas as pd

from typing import Dict

from utils.constants import SEED
from models.ps.ps import P_S_Network, loss

from utils.mind_dataset import MIND_Dataset, mind_collate_fn

torch.manual_seed(SEED)

def ps_train_loop(dataloader, model, loss_fn, optimizer, device):
    agg_loss = 0.0
    size = len(dataloader.dataset)
    for batch, (u, r, r_mask, s) in enumerate(dataloader):
        u, r, r_mask, s = u.to(device), r.to(device), r_mask.to(device), s.to(device)
        
        batch_sz = u.size(0)
        max_len = r.size(1)
        beta = torch.randn(batch_sz, max_len).to(device)
        out = model(u, r, r_mask, s, beta)
        loss = loss_fn(out, s)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        agg_loss += loss.item()
        if batch % (len(dataloader)//10) == 0:
            loss, current = loss.item() / batch_sz, batch * len(u)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return agg_loss / size
    
def train(
        ps_model: P_S_Network,
        train_df: pd.DataFrame, 
        user_d: Dict[str, int],
        item_d: Dict[str, int],
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 100):

    ps_save_path = 'ps_model.pth'
    
    ps_train_ds = MIND_Dataset(train_df, user_d, item_d)
    ps_train_dl = DataLoader(ps_train_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ps_model = ps_model.to(device)
 
    ps_optimizer = torch.optim.Adam(ps_model.parameters(), lr=learning_rate)

    for epoch_n in range(n_epochs):
        ps_train_loop(ps_train_dl, ps_model, loss, ps_optimizer, device)
        if epoch_n % 3 == 0:
            torch.save(ps_model.state_dict(), ps_save_path)

    torch.save(ps_model.state_dict(), ps_save_path)