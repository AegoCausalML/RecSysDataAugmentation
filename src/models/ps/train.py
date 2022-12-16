import torch

from utils.constants import SEED

torch.manual_seed(SEED)

def train_loop(dataloader, model, loss_fn, optimizer, device):
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
    