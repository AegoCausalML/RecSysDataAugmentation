import random
import pandas as pd
from torch.utils.data import Dataset

class P_Re_Dataset(Dataset):
    def __init__(self, mind_ds):
        """
        Negative sampling
        mind_ds: MIND_Dataset
        """
        super().__init__()
        self.mind_ds = mind_ds
        self.item_set = set(mind_ds.item_d.values())
        
        df_d = {'u':[], 'pos_i':[], 'neg_i':[]}
        for i in range(len(mind_ds)):
            u = mind_ds[i][0]
            pos_r = mind_ds[i][1]
            neg_r = random.sample(self.item_set - set(pos_r), len(pos_r))
            for pos_i, neg_i in zip(pos_r, neg_r):
                df_d['u'].append(u)
                df_d['pos_i'].append(pos_i)
                df_d['neg_i'].append(neg_i)
                
        self.df = pd.DataFrame(df_d)
    
    def __getitem__(self, idx):
        return self.df['u'].iloc[idx], self.df['pos_i'].iloc[idx], self.df['neg_i'].iloc[idx]
    
    def __len__(self):
        return len(self.df)