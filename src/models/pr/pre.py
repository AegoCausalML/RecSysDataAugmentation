import torch
import torch.nn as nn
import torch.nn.functional as F

class P_Re_Network(nn.Module):
    def __init__(self, n_users, n_itens, emb_dim=32):
        super(P_Re_Network, self).__init__()
        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_embeddings=n_itens, embedding_dim=emb_dim, padding_idx=0)
        self.w = nn.Parameter(torch.randn(n_itens))
        
    def forward(self, u, i, alpha):
        """
        Entrada:
            u: Indice do usuário
                torch.LongTensor(batch_sz)
            i: Indice do item
                torch.LongTensor(batch_sz)
            alpha: Variavel exogena correspondente aos itens em r
                torch.tensor(batch_sz)
        """
        # Seleciona os w correspondentes aos itens em r
        w_mul = alpha * self.w[i]

        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        emb_sim = (u_e * i_e).sum(dim=-1)

        return w_mul + emb_sim
    
def pre_loss_pos(pos_scores):
    return -F.logsigmoid(pos_scores).sum()

def pre_loss_neg(neg_scores):
    eps=1e-7
    return -torch.log(eps + 1 - torch.sigmoid(neg_scores)).sum()


def loss(pos_scores, neg_scores):
    """
    Negative sampling loss.
    Entrada:
        pos_scores: torch.tensor(batch_sz, pos_max_len)
        neg_scores: torch.tensor(batch_sz, neg_max_len)
    """
    # Loss to be minimized
    eps=1e-7
    pos_soft = F.logsigmoid(pos_scores).sum()
    neg_soft = torch.log(eps + 1 - torch.sigmoid(neg_scores)).sum()
    return -(pos_soft + neg_soft)