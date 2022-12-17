from typing import Dict

import torch

from models.ps.ps import P_S_Network as PS
from models.pr.pr import P_R_Network as PR
from models.cpr.cpr import gen_r_from_tau, gen_s
from models.gaussian_policy.gaussian_policy import GaussianPolicy
from models.gaussian_policy.recommender import Recommender

from utils.constants import SEED

torch.manual_seed(SEED)

def compute_reward(ps: PS,
                    Q: torch.tensor,
                    wR: torch.tensor,
                    recommender: Recommender,
                    taus: torch.tensor, 
                    item_idx_to_code: Dict[int, str],
                    user_idx_to_code: Dict[int, str],
                    u: torch.tensor, 
                    n_items: int, 
                    K: int, M: int
                ) -> torch.tensor:
    '''
    Recebe um batch de acoes e devolve um batch de recompensas
    
    Args:
        item_idx_to_code: Dicionário cuja chave é o índice de um item e o valor é seu código no dataset.
        user_idx_to_code: Dicionário cuja chave é o índice de um user e o valor é seu código no dataset.
        recommender: Recomendador usado para o treinamento da política gaussiana.
        taus: Um tensor de dimensões batch_size x dR, onde batch_size é o tamanho do batch e dR é a dimensão dos embeddings de p_r.
        u: Um tensor representado um batch de usuários, portanto, tem tamanho batch_size.
        K: Tamanho do conjunto de recomendações.
        M: Tamanho do conjunto de selecionados de R.
    
    Returns:
        Devolve um batch de recompensas.
    '''

    batch_sz = taus.size(0)
    reward = torch.zeros(batch_sz)

    for i in range(batch_sz):
        u_idx = u[i].item()
        tau = taus[i].unsqueeze(0)
        
        r = gen_r_from_tau(tau, Q, wR, K, n_items)
        s = gen_s(ps, u_idx, r, M)

        r = [item_idx_to_code[r_] for r_ in r]
        s = [item_idx_to_code[s_] for s_ in s]
        u_code = user_idx_to_code[u_idx]
        
        reward[i] = recommender.calculate_loss(u_code, r, s)

        print(f'u: {u_code} \n\tr: {r} \n\ts: {s}\n\treward: {reward[i]}')

    return reward

def reinforce_batch(policy: GaussianPolicy, 
                    optimizer: torch.optim.Optimizer, 
                    recommender: Recommender,
                    Q: torch.tensor,
                    wR: torch.tensor, 
                    ps: PS,
                    u: torch.tensor,
                    item_idx_to_code: Dict[int, str],
                    user_idx_to_code: Dict[int, str],
                    n_items: int,
                    K: int, M: int) -> torch.tensor:
    '''
    Treina a política em um batch de usuários e retorna a loss.

    Args:
        policy: Política gaussiana a ser aprendida.
        optimizer: Otimizador.
        recommender: Recomendador usado para o treinamento da política gaussiana.
        item_idx_to_code: Dicionário cuja chave é o índice de um item e o valor é seu código no dataset.
        user_idx_to_code: Dicionário cuja chave é o índice de um user e o valor é seu código no dataset.
        u: Um tensor representado um batch de usuários, portanto, tem tamanho batch_size.
        K: Tamanho do conjunto de recomendações.
        M: Tamanho do conjunto de selecionados de R.
    
    Returns:
        Retorna a loss do treino do batch.

    '''
    
    taus, log_prob = policy.act(u)
    
    reward = compute_reward(ps, Q, wR, recommender, taus, 
                            item_idx_to_code, user_idx_to_code, 
                            u, n_items, K, M)

    loss = -(log_prob * reward.unsqueeze(1)).sum() #Ideia: torch.ones(3,4) * torch.tensor([1,2,3]).unsqueeze(1)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(
        mind_path: str,
        pr: PR,
        ps: PS,
        n_users: int,
        n_items: int,
        item_idx_to_code: Dict[int, str],
        user_idx_to_code: Dict[int, str],
        K: int,
        M: int,
        dR: int,
        learning_rate: float = 1e-3,
        hidden_dimension: int = 16,
        batch_size: int = 16,
        n_episodes: int = 100):
    '''
    Treina a política gaussiana.

    Args:
        n_users: Número de usuários.
        n_items: Número de items.
        item_idx_to_code: Dicionário cuja chave é o índice de um item e o valor é seu código no dataset.
        user_idx_to_code: Dicionário cuja chave é o índice de um user e o valor é seu código no dataset.
        K: Tamanho da lista de recomendação.
        M: Tamanho do conjunto de selecionados de R.
        dR: Dimensão dos embeddins gerados por p_r. 
        hidden_dimension: número de neurôneos na camada oculta da política gaussiana.
        learning_rate: Taxa de aprendizado da política gaussiana.
        rec_model_path: Local onde está o modelo do recomendador pré-treinado.
        batch_size: Tamanho do batch para o treinamento da política gaussiana.
        n_episodes: Número de episódios para treinar a política gaussiana.
    
    '''
    
    Q = pr.item_emb.weight.data
    wR = pr.w.data.squeeze()


    print()
    print('-' * 15 + ' TREINANDO RECOMENDADOR PARA POLÍTICA GAUSSIANA ' + '-' * 15)
    print()

    recommender = Recommender(mind_path=mind_path)

    policy = GaussianPolicy(n_users, dR, hidden_dimension)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    for _ in range(n_episodes):
        u = torch.randint(1, n_users+1, (batch_size,))
        reinforce_batch(policy, 
                        optimizer, 
                        recommender,
                        Q, wR, ps, u, 
                        item_idx_to_code, 
                        user_idx_to_code, 
                        n_items,K, M)
    return policy