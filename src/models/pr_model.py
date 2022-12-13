import torch
import torch.nn as nn
import torch.nn.functional as F

class P_R_Network(nn.Module):
    def __init__(self, n_users, n_itens, emb_dim=32):
        super(P_R_Network, self).__init__()
        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_embeddings=n_itens, embedding_dim=emb_dim, padding_idx=0)
        self.w = nn.Parameter(torch.randn(1, n_itens))
        
    def forward(self, u, r, r_mask, alpha):
        """
        Entrada:
            u: Indice do usuário
                torch.LongTensor(batch_sz, 1)
            r: Tensor de indices dos itens recomendados ao usuário
                torch.LongTensor(batch_sz, max_len)
            r_mask: Máscara binária indicando quais elementos de r não são padding
                torch.LongTensor(batch_sz, max_len)
            alpha: Variavel exogena correspondente aos itens em r
                torch.tensor(batch_sz, max_len)
        """
        # Seleciona os w correspondentes aos itens em r
        w_mul = alpha * (self.w[0,r])

        u_e = self.user_emb(u)
        i_e = self.item_emb(r)
        emb_sim = torch.bmm(u_e, i_e.permute(0,2,1)).squeeze()

        return (w_mul + emb_sim) * r_mask

def p_r_loss(pos_scores, neg_scores):
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