import pandas as pd
import os
import random

from .constants import SEED

random.seed(SEED)

def data_to_recommended_and_selected(dataframe: pd.DataFrame, user_col: str = 'UserID', interactions_col: str = 'Impressions'):
    data = list()
    for user, user_interactions in zip(dataframe[user_col].values, dataframe[interactions_col].values):
        # Interações são da forma noticia_id-clicado, exemplo: N23699-0 N21291-0 N1901-1 N27292-0 N17443-0
        R = [interaction.split('-')[0] for interaction in user_interactions.split()] 
        S = [interaction.split('-')[0] for interaction in user_interactions.split() if interaction[-1] == '1']
        data.append((user, ' '.join(R), ' '.join(S)))
    return pd.DataFrame(data, columns=['UserID', 'R', 'S'])

def gen_code_dict(code_l):
    #gen_code_dict(list(df['UserID'].unique()))
    #gen_code_dict(" ".join(df['R']).split(" "))
    code_d = {}
    i = 1 # 0 eh PADDING
    for code in set(code_l):
        code_d[code] = i
        i += 1
    return code_d

def preprocess_mind(data_path = 'mind', out_path = 'data'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Coleta dos dados
    train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
    valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

    behaviors_col_names = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']

    train_data = pd.read_table(train_behaviors_file, header=None, names=behaviors_col_names)
    valid_data = pd.read_table(valid_behaviors_file, header=None, names=behaviors_col_names)

    train_data = data_to_recommended_and_selected(train_data)
    valid_data = data_to_recommended_and_selected(valid_data)

    # Pre-processamento
    df_ids = list(range(len(train_data)))
    random.shuffle(df_ids)
    train_ids = df_ids[:int(len(train_data)*0.9)]
    val_ids = df_ids[int(len(train_data)*0.9):]

    train_df = train_data.iloc[train_ids]
    val_df = train_data.iloc[val_ids]

    user_d = gen_code_dict(list(train_df['UserID'].unique()))
    item_d = gen_code_dict(" ".join(train_df['R']).split(" "))

    item_check_s = val_df.apply(lambda x: x['R'].split() + x['S'].split(), axis=1).apply(lambda l: all([ i in item_d for i in l]))
    user_check_s = val_df['UserID'].apply(lambda x: x in user_d)
    val_df = val_df[item_check_s & user_check_s]

    # Salvar datasets
    train_df.to_csv(os.path.join(out_path, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(out_path, 'val_df.csv'), index=False)
    
    # Salvar os dicts de usuario e item
    pd.DataFrame({'code':user_d.keys(), 'indice':user_d.values()}).to_csv(os.path.join(out_path, 'user_d.csv'), index=False)
    pd.DataFrame({'code':item_d.keys(), 'indice':item_d.values()}).to_csv(os.path.join(out_path, 'item_d.csv'), index=False)
