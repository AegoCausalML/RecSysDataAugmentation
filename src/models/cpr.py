import torch

from utils.constants import SEED

torch.manual_seed(SEED)

class CPR:
    def __init__(self, pr_model, ps_model, n_items):
        self.pr_model = pr_model
        self.Q = pr_model.item_emb.weight.data
        self.wR = pr_model.w.data.squeeze()
        
        self.ps_model = ps_model
        self.n_items = n_items
    
    def sample_alpha_posterior(self):
        return torch.randn(self.n_items)
    
    def sample_beta_posterior(self, k):
        return torch.randn(k)
    
    def gen_r_from_tau(self, tau, k):
        """
        tau: action center - torch.tensor(1, emb_dim)
        k: number of items - int
        """
        alpha = self.sample_alpha_posterior()
        score = (tau @ self.Q.T) + (self.wR * alpha)

        scores_dict = dict(enumerate(score[0]))
        sorted_dict = dict(sorted(scores_dict.items(), key=lambda item:item[1], reverse=True))

        return list(sorted_dict.keys())[:k]

    def gen_s(self, u, r, M):
        """
        u: user id - int
        r: list of item ids - [int]
        M - itens to be selected - int
        """
        beta = self.sample_beta_posterior(len(r))
        
        u_b = torch.LongTensor([[u]])
        r_b = torch.LongTensor([r])
        r_mask_b = torch.ones(1, len(r))
        beta_b = beta.unsqueeze(0)
        score = self.ps_model(u_b, r_b, r_mask_b, None, beta_b).data[0]
        
        scores_dict = dict(zip(r, score))
        sorted_dict = dict(sorted(scores_dict.items(), key=lambda item:item[1], reverse=True))
        return list(sorted_dict.keys())[:M]