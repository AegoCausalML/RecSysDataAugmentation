import os
from tqdm import tqdm
import pandas as pd
import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

# tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

# based on: https://github.com/microsoft/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

class NRMS:
    def __init__(self, model_path=None, mind_type='small', test_src='train', seed=42) -> None:
        self.mind_type = mind_type

        self.download_resources()

        self.hparams =  prepare_hparams(
                            self.yaml_file, 
                            wordEmb_file=self.wordEmb_file,
                            wordDict_file=self.wordDict_file, 
                            userDict_file=self.userDict_file,
                            batch_size=1,
                            epochs=1
                        )  

        self.interator = MINDIterator
        self.model = NRMSModel(self.hparams, self.interator, seed=seed)

        if test_src == 'train':
            test_behaviors_file = self.train_behaviors_file 
            test_news_file = self.train_news_file

        else:
            test_behaviors_file = self.valid_behaviors_file 
            test_news_file = self.valid_news_file

        behaviors_col_names = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        news_col_names = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']

        self.test_behaviors_src = pd.read_table(test_behaviors_file, header=None, names=behaviors_col_names)
        self.test_news_src = pd.read_table(test_news_file, header=None, names=news_col_names)

        # se model é diferente de None, carregá-lo sem precisar treinar
        if model_path:
            self.model.model.load_weights(model_path)

    def train(self, train_news_file=None, train_behaviors_file=None, valid_news_file=None, valid_behaviors_file=None):
        train_behaviors_file = train_behaviors_file if train_behaviors_file else self.train_behaviors_file
        train_news_file = train_news_file if train_news_file else self.train_news_file
        
        valid_behaviors_file = valid_behaviors_file if valid_behaviors_file else self.valid_behaviors_file
        valid_news_file = valid_news_file if valid_news_file else self.valid_news_file

        self.model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

    def test_loss(self):
        iterator = self.interator(self.hparams, npratio=4, col_spliter='¨',)
        tqdm_util = tqdm(iterator.load_data_from_file(self.counterfactual_news_file, self.counterfactual_behaviors_file),  disable=True)

        loss = 0
        step = 0

        for batch_data_input in tqdm_util:
            test_input, test_label = self.model._get_input_label_from_iter(batch_data_input)
            step_loss = self.model.model.test_on_batch(test_input, test_label)

            loss += step_loss
            step += 1

        return loss / step

    def calculate_loss(self, u, R, S):
        counterfactual_impression = ' '.join(list(map(lambda x: f'{x}-{1 if x in S else 0}', R)))
        counterfactual_behavior = self.test_behaviors_src[self.test_behaviors_src.UserID == u]

        counterfactual_behavior.Impressions = counterfactual_behavior.Impressions.apply(lambda x: counterfactual_impression) 
        
        counterfactual_history = counterfactual_behavior.iloc[0].History

        # print(f'History: {counterfactual_history}')
        if type(counterfactual_history) is str:
            news_id_in_counterfactual = set(counterfactual_history.split() + R)

        else:
            news_id_in_counterfactual = set(R)

        news_in_counterfactual = self.test_news_src.query(f'NewsID in {tuple(news_id_in_counterfactual)}') 

        counterfactual_behavior.to_csv(self.counterfactual_behaviors_file, sep='¨', index=False, header=None)
        news_in_counterfactual.to_csv(self.counterfactual_news_file, sep='¨', index=False, header=None)

        return self.test_loss()
    

    def download_resources(self):
        # tmpdir = TemporaryDirectory()
        
        self.data_path = './nrms_data/'

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.train_news_file = os.path.join(self.data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(self.data_path, 'train', r'behaviors.tsv')
        self.valid_news_file = os.path.join(self.data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(self.data_path, 'valid', r'behaviors.tsv')
        self.wordEmb_file = os.path.join(self.data_path, 'utils', 'embedding.npy')
        self.userDict_file = os.path.join(self.data_path, 'utils', 'uid2index.pkl')
        self.wordDict_file = os.path.join(self.data_path, 'utils', 'word_dict.pkl')
        self.yaml_file = os.path.join(self.data_path, 'utils', r'nrms.yaml')

        mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(self.mind_type)

        counterfactual_path = os.path.join(self.data_path, 'counterfactual')

        self.counterfactual_news_file = os.path.join(counterfactual_path, r'news.tsv')
        self.counterfactual_behaviors_file = os.path.join(counterfactual_path, r'behaviors.tsv')

        if not os.path.exists(counterfactual_path):
            os.makedirs(counterfactual_path)

        if not os.path.exists(self.train_news_file):
            download_deeprec_resources(mind_url, os.path.join(self.data_path, 'train'), mind_train_dataset)
            
        if not os.path.exists(self.valid_news_file):
            download_deeprec_resources(mind_url, \
                                    os.path.join(self.data_path, 'valid'), mind_dev_dataset)
        if not os.path.exists(self.yaml_file):
            download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                                    os.path.join(self.data_path, 'utils'), mind_utils) 

if __name__ == '__main__':
    nrms = NRMS(model_path='./nrms_data/model/nrms_ckpt')
    loss = nrms.calculate_loss('U78413', ['N6193', 'N54013', 'N55912', 'N56774', 'N43214'],['N43214', 'N55912'])
    print(loss)