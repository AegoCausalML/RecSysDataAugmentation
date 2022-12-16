import random
import torch
from torch.utils.data import Dataset

from utils.constants import SEED

random.seed(SEED)
torch.manual_seed(SEED)

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