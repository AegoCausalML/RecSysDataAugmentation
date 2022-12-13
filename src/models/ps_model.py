import torch
import torch.nn as nn
import torch.nn.functional as F

class P_S_Network(nn.Module):
    def __init__(self, n_users, n_itens, K, emb_dim=32):
        super(P_S_Network, self).__init__()
        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_embeddings=n_itens, embedding_dim=emb_dim, padding_idx=0)
        self.w = nn.Parameter(torch.randn(1, K))
        self.K = K
        
    def forward(self, u, r, r_mask, s, beta):
        """
        Entrada:
            u: Indice do usuário
                torch.LongTensor(batch_sz, 1)
            r: Tensor de indices dos itens recomendados ao usuário
                torch.LongTensor(batch_sz, max_len)
            r_mask: Máscara binária indicando quais elementos de r não são padding
                torch.LongTensor(batch_sz, max_len)
            s: Máscara binária indicando quais elementos de r estão em s.
                torch.LongTensor(batch_sz, max_len)
            beta: Variavel exogena correspondente aos itens em r
                torch.tensor(batch_sz, max_len)
        """
        max_len = r.size(1)
        w_mul = beta * (self.w[:,:max_len])

        u_e = self.user_emb(u)
        i_e = self.item_emb(r)
        emb_sim = torch.bmm(u_e, i_e.permute(0,2,1)).squeeze()

        return (w_mul + emb_sim) * r_mask

def p_s_loss(out, s):
    # Loss to be minimized
    out_soft = F.log_softmax(out, dim=1)
    return -(out_soft*s).sum()