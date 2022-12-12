import torch

from ..utils.constants import SEED

class CPR_Env:
    def __init__(self, cpr, recommender, item_idx_to_code, user_idx_to_code, k, M):
        self.cpr = cpr
        self.recommender = recommender
        self.k = k
        self.M = M
        self.item_idx_to_code = item_idx_to_code
        self.user_idx_to_code = user_idx_to_code

    def compute_reward(self, action, u):
        """
        Recebe um batch de acoes e devolve um batch de recompensas
        action: torch.tensor(batch_sz, emb_dim)
        u: torch.tensor(batch_sz)
        reward: torch.tensor(batch_sz)
        """
        batch_sz = action.size(0)
        reward = torch.zeros(batch_sz)
        for i in range(batch_sz):
            u_id = u[i].item()
            tau = action[i].unsqueeze(0)
            
            r = self.cpr.gen_r_from_tau(tau, self.k)
            s = self.cpr.gen_s(u_id, r, self.M)

            r = [self.item_idx_to_code[r_] for r_ in r]
            s = [self.item_idx_to_code[s_] for s_ in s]
            u_code = self.user_idx_to_code[u_id]
            
            reward[i] = self.recommender.calculate_loss(u_code, r, s)
            print(f'u: {u_code} \n\tr: {r} \n\ts: {s}\n\treward: {reward[i]}')
        
        return reward