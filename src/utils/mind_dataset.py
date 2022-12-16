import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.constants import SEED

torch.manual_seed(SEED)

class MIND_Dataset(Dataset):
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