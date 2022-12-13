import torch
from torch.utils.data import Dataset

import pandas as pd
import random

# P_S
class MIND_dataset(Dataset):
    def get_picked_inds(self, r, s):
        ind_l = []
        for picked_i in s:
            for ind, i in enumerate(r):
                if i == picked_i:
                    ind_l.append(ind)
                    break
        return ind_l

    def __init__(self, df, user_d, item_d):
        """
        Recebe um DataFrame com as colunas ['UserID', 'R', 'S']
        UserID: ids de usuario
        R: listas de ids de itens recomendados a U
        S: listas de ids de itens de R escolhidos por U
        
        user_d: dicionario que mapeia de id de usuário para numero inteiro, sendo que 0 é PAD
        item_d: dicionario que mapeia de id de item para numero inteiro, sendo que 0 é PAD
        """
        super().__init__()
        # Transformar os codigos de item e usuario em indices inteiros. 0 eh PADDING
        self.user_d = user_d
        self.item_d = item_d

        self.df = pd.DataFrame(index=df.index)
        self.df['U'] = df['UserID'].apply(lambda x: self.user_d[x])
        self.df['R'] = df['R'].apply(lambda x: [self.item_d[i] for i in x.split(' ')])
        # S
        self.df['S_picked'] = df['S'].apply(lambda x: [self.item_d[i] for i in x.split(' ')])
        self.df['S'] = self.df.apply(lambda x: self.get_picked_inds(x['R'], x['S_picked']), axis=1)
        
    def __getitem__(self, idx):
        '''
        Retorna uma tupla com U, R e S
        U => id do usuário
        R => lista de ids de itens
        S => lista de indices de R indicando quais itens foram escolhidos.
        '''
        return self.df['U'].iloc[idx], self.df['R'].iloc[idx], self.df['S'].iloc[idx]
    
    def __len__(self):
        return len(self.df)

    def get_n_users(self):
        return len(self.user_d)+1

    def get_n_items(self):
        return len(self.item_d)+1

def mind_collate_fn(data):
    '''
    Retorna tupla: u, r, r_mask, s
    '''
    batch_sz = len(data)
    max_len = max([len(r) for _,r,_ in data])

    u_batch = torch.zeros(batch_sz, 1, dtype=torch.long)
    r_batch = torch.zeros(batch_sz, max_len, dtype=torch.long)
    r_mask_batch = torch.zeros(batch_sz, max_len, dtype=torch.long)
    s_batch = torch.zeros(batch_sz, max_len, dtype=torch.long)

    for i, (u,r,s) in enumerate(data):
        u_batch[i] = u
        r_batch[i] = torch.LongTensor(r + [0]*(max_len-len(r)))
        r_mask_batch[i] = torch.LongTensor([1]*len(r) + [0]*(max_len-len(r)))
        s_batch[i] = torch.LongTensor([(1 if ind in s else 0) for ind in range(max_len)])
    return u_batch, r_batch, r_mask_batch, s_batch


# P_R
class MIND_P_R_Dataset(Dataset):
    def __init__(self, mind_ds, max_sampling_len):
        """
        Retorna amostras com negative sampling.
        mind_ds: MIND_Dataset
        """
        super().__init__()
        self.mind_ds = mind_ds
        self.item_set = set(mind_ds.item_d.values())
        self.max_sampling_len = max_sampling_len
    
    def __getitem__(self, idx):
        '''
        Retorna uma tupla com U, R_pos, R_neg
        U => id do usuário
        R_pos => lista de ids de itens em r para U amostrados
        R_neg => lista de ids de itens não em r para U amostrados
        '''
        u,r,_ = self.mind_ds[idx]
        pos_r = random.sample(r, min(self.max_sampling_len, len(r)))
        neg_r = random.sample(self.item_set - set(r), len(pos_r))
        return u, pos_r, neg_r
    
    def __len__(self):
        return len(self.mind_ds)
    
def mind_p_r_collate_fn(data):
    '''
    Retorna tupla: u, pos_r, pos_r_mask, neg_r, neg_r_mask
    '''
    batch_sz = len(data)
    pos_max_len = max([len(pos_r) for _,pos_r,_ in data])
    neg_max_len = max([len(neg_r) for _,_,neg_r in data])

    u_batch = torch.zeros(batch_sz, 1, dtype=torch.long)
    pos_r_batch = torch.zeros(batch_sz, pos_max_len, dtype=torch.long)
    pos_r_mask_batch = torch.zeros(batch_sz, pos_max_len, dtype=torch.long)
    neg_r_batch = torch.zeros(batch_sz, neg_max_len, dtype=torch.long)
    neg_r_mask_batch = torch.zeros(batch_sz, neg_max_len, dtype=torch.long)

    for i, (u, pos_r, neg_r) in enumerate(data):
        u_batch[i] = u
        pos_r_batch[i] = torch.LongTensor(pos_r + [0]*(pos_max_len-len(pos_r)))
        pos_r_mask_batch[i] = torch.LongTensor([1]*len(pos_r) + [0]*(pos_max_len-len(pos_r)))
        neg_r_batch[i] = torch.LongTensor(neg_r + [0]*(neg_max_len-len(neg_r)))
        neg_r_mask_batch[i] = torch.LongTensor([1]*len(neg_r) + [0]*(neg_max_len-len(neg_r)))
        
    return u_batch, pos_r_batch, pos_r_mask_batch, neg_r_batch, neg_r_mask_batch