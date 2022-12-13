from scm_datasets import MIND_P_R_Dataset, mind_p_r_collate_fn, MIND_dataset
from pr_model import P_R_Network

import os
from os.path import join
import pandas as pd
import torch
from torch.utils.data import DataLoader

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

def pr_test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for u, pos_r, pos_r_mask, neg_r, neg_r_mask in dataloader:
            u, pos_r, pos_r_mask, neg_r, neg_r_mask = u.to(device), pos_r.to(device), pos_r_mask.to(device), neg_r.to(device), neg_r_mask.to(device)
            batch_sz = u.size(0)
            pos_max_len = pos_r.size(1)
            neg_max_len = neg_r.size(1)
        
            pos_alpha = torch.randn(batch_sz, pos_max_len).to(device)
            pos_scores = model(u, pos_r, pos_r_mask, pos_alpha)
        
            neg_alpha = torch.randn(batch_sz, neg_max_len).to(device)
            neg_scores = model(u, neg_r, neg_r_mask, neg_alpha)
            
            test_loss += loss_fn(pos_scores, neg_scores).item() / batch_sz

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    return test_loss


def load_checkpoint(save_path):
    """
    Retorna modelo e log_loss
    """
    model = 
    if not os.path.exists(save_path):
        os.makedirs(save_path)



if __name__ == "__main__":
    """
    Treina P_R, gerando logs de loss por epoca e checkpoint (state_dict) do modelo.
    Recomeça de onde parou caso já tenha um checkpoint.
    """
    # Parâmetros:
    data_path = 'MIND-small_pp' # Saída do preprocess_mind.py
    save_path = 'pr_model'

    end_epoch = 40
    batch_size = 32
    learning_rate = 1e-3
    p_r_max_sampling_len = 5

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Carrega dados
    train_df = pd.read_csv(join(data_path, 'train_df.csv'))
    val_df = pd.read_csv(join(data_path, 'val_df.csv'))

    user_df = pd.read_csv(join(data_path, 'user_d.csv'))
    user_d = {code: ind for code, ind in zip(user_df['code'], user_df['indice'])}
    item_df = pd.read_csv(join(data_path, 'item_d.csv'))
    item_d = {code: ind for code, ind in zip(item_df['code'], item_df['indice'])}

    # Cria dataset e dataloaders
    ps_train_ds = MIND_dataset(train_df, user_d, item_d)
    ps_val_ds = MIND_dataset(val_df, user_d, item_d)

    train_ds = MIND_P_R_Dataset(ps_train_ds, p_r_max_sampling_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_p_r_collate_fn)

    val_ds = MIND_P_R_Dataset(ps_val_ds, p_r_max_sampling_len)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_p_r_collate_fn)

    #Modelo
    #Checar checkpoint
    pr_save_path = 'pr_model.pth'
    pr_loss_save_path = 'pr_loss.csv'
    pr_loss_d = {'epoch': [], 'train':[], 'val':[]}

    model = P_R_Network(len(user_d)+1, len(item_d)+1)

    # Treinamento
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    model = model.to(device)
    print('P_R model parameters:')     
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())
            print('Cuda = ', param.is_cuda)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)