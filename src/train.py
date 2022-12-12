import os
from os.path import join

import torch
import pandas as pd

from models.gaussian_policy import GaussianPolicy
from models.recommenders.nrms import NRMS
from models.cpr import CPR
from models.pr import P_R_Network, pr_train_loop
from models.ps import P_S_Network, ps_train_loop
from datasets.scm_datasets import gen_code_dict
from utils.preprocess_mind import preprocess_mind

def main(pr_model_path='pr_model.pth',
        mind_path='../data/mind/',
        mind_processed_path='../data/mind/processed/'):
    
    print('-' * 15 + ' PREPROCESSING MIND ' + '-' * 15)
    preprocess_mind(mind_path, mind_processed_path)

    print('-' * 15 + ' TRAINING PR ' + '-' * 15)

    user_df = pd.read_csv(join(mind_processed_path, 'user_d.csv'))
    user_d = {code: ind for code, ind in zip(user_df['code'], user_df['indice'])}
    user_idx_to_code = {ind: code for code, ind in zip(user_df['code'], user_df['indice'])}

    item_df = pd.read_csv(join(mind_processed_path, 'item_d.csv'))
    item_d = {code: ind for code, ind in zip(item_df['code'], item_df['indice'])}
    item_idx_to_code = {ind: code for code, ind in zip(item_df['code'], item_df['indice'])}

    batch_size = 32
    p_r_max_sampling_len = 5

    ps_train_ds = MIND_dataset(train_df, user_d, item_d)
    ps_train_dl = DataLoader(ps_train_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_collate_fn)

    ps_val_ds = MIND_dataset(val_df, user_d, item_d)
    ps_val_dl = DataLoader(ps_val_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_collate_fn)


    pr_train_ds = MIND_P_R_Dataset(ps_train_ds, p_r_max_sampling_len)
    pr_train_dl = DataLoader(pr_train_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_p_r_collate_fn)

    pr_val_ds = MIND_P_R_Dataset(ps_val_ds, p_r_max_sampling_len)
    pr_val_dl = DataLoader(pr_val_ds, batch_size=batch_size, shuffle=True, collate_fn=mind_p_r_collate_fn)
    print('-' * 15 + ' TRAINING PS ' + '-' * 15)

    # print('-' * 15 + ' TRAINING PS ' + '-' * 15)
    
    # print('-' * 15 + ' TRAINING GAUSSIAN POLICY ' + '-' * 15)
    
    # gaussian_policy = GaussianPolicy(n_users, dR, hidden_dimension) 

    # pr_model = P_R_Network(len(user_d)+1, len(item_d)+1, emb_dim=2)
    
    # if os.path.exists(pr_model_path):    
    #     pr_model.load_state_dict(torch.load(pr_model_path))

    # else:
    #     pr_model.train()


if __name__ == '__main__':
    main()