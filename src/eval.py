import os 

from models.recommenders.nrms import NRMS
from models.recommenders.lstur import LSTUR

if __name__ == '__main__':
    data_path = './mind'

    train_iterator = {
        'factual': (os.path.join(data_path, 'train', 'news.tsv'), os.path.join(data_path, 'train', 'behaviors.tsv')),
        'contrafactual': (os.path.join(data_path, 'train', 'news.tsv'), os.path.join(data_path, 'counterfactual', 'behaviors.tsv')),
        'factual com contrafactual': (os.path.join(data_path, 'train', 'news.tsv'), os.path.join(data_path, 'factual_with_counterfactual', 'behaviors.tsv'))
    }

    valid_news_file = os.path.join(data_path, 'valid', 'news.tsv')
    valid_behaviors_file = os.path.join(data_path, 'valid', 'behaviors.tsv')

    models = {
        'NRMS': NRMS,
        'LSTUR': LSTUR
    }

    for model_name, model in models.items():
        m = model()
        for data_type, (train_news_file, train_behaviors_file) in train_iterator.items():
            performance = m.train(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
            
            print('=' * 100)
            
            print(f'Performance para o modelo: {model_name}')
            print(f'\tTipo de dados: {data_type}')
            print(f'\t\t{performance}')

            print('=' * 100)