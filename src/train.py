# from os.path import join

import os
import random
from typing import Dict

import torch
import pandas as pd
from tqdm import tqdm

import argparse

# from torch.utils.data import DataLoader

# from models.gaussian_policy import GaussianPolicy
# from models.recommenders.nrms import NRMS
# from models.cpr import CPR
# from models.pr import P_R_Network, pr_train_loop
# from models.ps import P_S_Network, ps_train_loop
# from datasets.scm_datasets import gen_code_dict, MIND_dataset, MIND_P_R_Dataset, mind_collate_fn, mind_p_r_collate_fn
# from utils.preprocess_mind import preprocess_mind

from models.pr.pr import P_R_Network as PR
from models.pr.train import train as pr_train
from models.ps.train import train as ps_train
from models.ps.ps import P_S_Network as PS
from models.gaussian_policy.gaussian_policy import GaussianPolicy
from models.gaussian_policy.train import train as train_policy
from models.cpr.cpr import gen_r_from_tau, gen_s

from utils.download_mind import download_mind
from utils.preprocess_mind import preprocess_mind
from utils.constants import SEED

torch.manual_seed(SEED)
random.seed(SEED)
    
def read_data(data_path: str = ''):

    user_df = pd.read_csv(os.path.join(data_path, 'user_d.csv'))
    user_d = {code: ind for code, ind in zip(user_df['code'], user_df['indice'])}
    user_idx_to_code = {ind: code for code, ind in zip(user_df['code'], user_df['indice'])}

    item_df = pd.read_csv(os.path.join(data_path, 'item_d.csv'))
    item_d = {code: ind for code, ind in zip(item_df['code'], item_df['indice'])}
    item_idx_to_code = {ind: code for code, ind in zip(item_df['code'], item_df['indice'])}

    return user_d, item_d, user_idx_to_code, item_idx_to_code

def load_pr_and_ps(n_users: int, n_items: int,  dR: int, dS: int, M: int,
                    model_path: str = 'saved_models/'):

    pr = PR(n_users + 1, n_items + 1, emb_dim=dR)
    pr.load_state_dict(torch.load(os.path.join(model_path, 'pr', 'pr_model.pth')))

    ps = PS(n_users+1, n_items + 1, M, emb_dim=dS)
    ps.load_state_dict(torch.load(os.path.join(model_path, 'ps', 'ps_model.pth')))

    return pr, ps 

def generate_counterfactual_impressions(ps: PS,
                                        policy: GaussianPolicy,
                                        Q: torch.tensor,
                                        wR: torch.tensor,
                                        base_behavior: pd.DataFrame,
                                        user_d: Dict[str, int],
                                        item_idx_to_code: Dict[int, str],
                                        user_idx_to_code: Dict[int, str], 
                                        n_items: int,
                                        n_users: int = 10000,
                                        batch_size: int = 16):
    data = list()

    users = list(user_d.values())[:n_users] 
    behavior_id = base_behavior.index.stop + 1

    for i in tqdm(range(0, len(users) + 1, batch_size)):
        try:
            u = torch.as_tensor(users[i:i+batch_size])

            taus, _ = policy.act(u)

            for i in range(len(u)):
                k = random.randint(1, 10)
                M = random.randint(0, k)

                u_idx = u[i].item()
                tau = taus[i].unsqueeze(0)
                
                r = gen_r_from_tau(tau, Q, wR, K, n_items)
                s = gen_s(ps, u_idx, r, M) if M > 0 else []

                r = [item_idx_to_code[r_] for r_ in r]
                s = [item_idx_to_code[s_] for s_ in s]

                u_code = user_idx_to_code[u_idx]

                counterfactual_impression = ' '.join(list(map(lambda x: f'{x}-{1 if x in s else 0}', r)))

                # print(type(base_behavior[base_behavior.UserID == u_code]))

                # counterfactual_behavior.Impressions = counterfactual_behavior.Impressions.apply(lambda x: counterfactual_impression) 
                for _, row in base_behavior[base_behavior.UserID == u_code].iterrows():
                    data.append((behavior_id, row.UserID, row.Time, row.History, counterfactual_impression ))
                    behavior_id +=1
        except:
            continue

    return pd.DataFrame(data, columns=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Pârametros gerais
    parser.add_argument('-k', action='store', dest='k', default=5, type=int, required=False,
                        help='GERAL: Tamanho da lista de recomendação de treino.')
    parser.add_argument('-mind_path', action='store', dest='mind_path', default='mind/', required=False,
                    help='GERAL: Pasta onde o dataset mind está ou será baixado.')
    parser.add_argument('-data_path', action='store', dest='data_path', default='data/', required=False,
                    help='GERAL: Pasta onde dados processados serão armazenados.')
    
    # Parâmetros de PR
    parser.add_argument('-pr_epochs', action='store', dest='pr_epochs', default=100, type=int, required=False,
                        help='PR: Número de épocas de treinamento.')
    parser.add_argument('-pr_lr', action='store', dest='pr_lr', default=1e-3, type=float, required=False,
                        help='PR: Taxa de treinamento.')
    parser.add_argument('-pr_batch_size', action='store', dest='pr_batch_size', default=32, type=int, required=False,
                        help='PR: Tamanho dos batches de treinameno.')
    parser.add_argument('-pr_max_sampling', action='store', dest='pr_max_sampling', default=5, type=int, required=False,
                        help='PR: Tamanho máximo de sampling.')
    parser.add_argument('-dr', action='store', dest='dr', default=2, type=int, required=False,
                        help='PR: Dimensão do embedding.')
    
    # Parâmetros de PS
    parser.add_argument('-ps_epochs', action='store', dest='ps_epochs', default=5, type=int, required=False,
                        help='PS: Número de épocas de treinamento de PS.')
    parser.add_argument('-ps_lr', action='store', dest='ps_lr', default=1e-3, type=float, required=False,
                        help='PS: Taxa de treinamento.')
    parser.add_argument('-ps_batch_size', action='store', dest='ps_batch_size', default=32, type=int, required=False,
                        help='PR: Tamanho dos batches de treinameno.')
    parser.add_argument('-ds', action='store', dest='ds', default=2, type=int, required=False,
                        help='PS: Dimensão do embedding.')
    parser.add_argument('-m_train', action='store', dest='m_train', default=299, type=int, required=False,
                        help='PS: Quantidade de selecionados da lista de recomendação de treino.')
    
    # Parâmetros da política gaussiana
    parser.add_argument('-m', action='store', dest='m', default=2, type=int, required=False,
                        help='Política Gaussiana: Quantidade de selecionados da lista de recomendação de treino.')

    parser.add_argument('-gp_hidden_dimension', action='store', dest='gp_hidden_dimension', default=16, type=int, required=False,
                        help='Política Gaussiana: Dimensão da camada oculta.')
    parser.add_argument('-gp_lr', action='store', dest='gp_lr', default=1e-3, type=float, required=False,
                        help='Política Gaussiana: Taxa de treinamento.')
    parser.add_argument('-gp_batch_size', action='store', dest='gp_batch_size', default=32, type=int, required=False,
                        help='Política Gaussiana: Tamanho dos batches de treinameno.')
    parser.add_argument('-gp_episodes', action='store', dest='gp_episodes', default=100, type=int, required=False,
                        help='Política Gaussiana: Número de episódios de treinamento.')

    arguments = parser.parse_args()
    
    dR = arguments.dr
    dS = arguments.ds
    M_train = arguments.m_train
    M = arguments.m
    K = arguments.k
    
    mind_path = arguments.mind_path
    data_path = arguments.data_path

    print()
    print('-' * 15 + ' BAIXANDO MIND ' + '-' * 15)
    print()
    
    download_mind(mind_path)

    print()
    print('-' * 15 + ' PRÉ-PROCESSANDO MIND ' + '-' * 15)
    print()
    preprocess_mind(mind_path, data_path)

    user_d, item_d, user_idx_to_code, item_idx_to_code = read_data('data/')
    n_users, n_items = len(user_d), len(item_d)

    print()
    print('-' * 15 + ' TREINANDO PR ' + '-' * 15)
    train_df = pd.read_csv('data/train_df.csv')
    pr = PR(n_users+1, n_items+1, emb_dim=dR)
    pr_train(pr,train_df, user_d, item_d,
            arguments.pr_max_sampling, arguments.pr_lr, 
            arguments.pr_batch_size, arguments.pr_epochs)
    print()

    print()
    print('-' * 15 + ' TREINANDO PS ' + '-' * 15)
    ps = PS(n_users+1, n_items + 1, M_train, emb_dim=dS)
    ps_train(ps, train_df, user_d, item_d, 
            arguments.ps_lr, arguments.ps_batch_size, arguments.ps_epochs)
    print()

    print()
    print('-' * 15 + ' TREINANDO POLÍTICA GAUSSIANA ' + '-' * 15)
    print()

    policy = train_policy(mind_path, pr, ps, n_users, n_items, item_idx_to_code, 
                        user_idx_to_code, K, M, dR, arguments.gp_lr, 
                        arguments.gp_hidden_dimension, arguments.gp_batch_size, arguments.gp_episodes)

    print()
    print('-' * 15 + ' GERANDO DADOS CONTRAFACTUAIS ' + '-' * 15)
    print()

    # Geração de dados contrafactuais
    behaviors_col_names = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
    base_behavior = pd.read_table(os.path.join(mind_path, 'train', 'behaviors.tsv'), 
                                    header=None, names=behaviors_col_names)

    Q = pr.item_emb.weight.data
    wR = pr.w.data.squeeze()

    # # Salva os dados contrafactuais gerados para serem usados na avaliação
    counterfactual_behaviors = generate_counterfactual_impressions(ps, policy, Q, wR, base_behavior, user_d, 
                                                                    item_idx_to_code, user_idx_to_code, n_items, n_users=10)

    counterfactual_behaviors.to_csv(os.path.join(mind_path, 'counterfactual', 'behaviors.tsv'), sep='\t', index=False, header=False)

    with open(os.path.join(mind_path, 'train', 'behaviors.tsv')) as f:
        factual_lines = f.readlines()

    with open(os.path.join(mind_path, 'counterfactual', 'behaviors.tsv')) as f:
        counterfactual_lines = f.readlines()
    
    counterfactual_path = os.path.join(mind_path, 'factual_with_counterfactual')
    if not os.path.exists(counterfactual_path):
        os.makedirs(counterfactual_path)

    factual_with_counterfactual_file = os.path.join(counterfactual_path, 'behaviors.tsv')
    with open(factual_with_counterfactual_file, 'w') as f:
        factual_with_counterfactual_lines = factual_lines + counterfactual_lines
        for line in factual_with_counterfactual_lines:
            f.write(line)