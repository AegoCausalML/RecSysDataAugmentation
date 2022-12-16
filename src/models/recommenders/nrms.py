import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

from utils.constants import SEED

# Baseado em: https://github.com/microsoft/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

class NRMS:
    def __init__(self, mind_type: str = 'small', data_path: str = './mind', epochs: int = 5, batch_size: int = 32) -> None:
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_news_file = os.path.join(data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')

        self.valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

        self.wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
        self.userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
        self.wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")

        self.yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

        mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)

        if not os.path.exists(self.train_news_file):
            download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
            
        if not os.path.exists(self.valid_news_file):
            download_deeprec_resources(mind_url, \
                                    os.path.join(data_path, 'valid'), mind_dev_dataset)
        if not os.path.exists(self.yaml_file):
            download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                                    os.path.join(data_path, 'utils'), mind_utils)

        self.hparams = prepare_hparams(self.yaml_file, 
                          wordEmb_file=self.wordEmb_file,
                          wordDict_file=self.wordDict_file, 
                          userDict_file=self.userDict_file,
                          batch_size=self.batch_size,
                          epochs=self.epochs)

        self.iterator = MINDIterator

    def train(self, train_news_file: str, train_behaviors_file: str, valid_news_file: str, valid_behaviors_file: str):
        model = NRMSModel(self.hparams, self.iterator, seed=SEED)
        model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
        return model.run_eval(valid_news_file, valid_behaviors_file)