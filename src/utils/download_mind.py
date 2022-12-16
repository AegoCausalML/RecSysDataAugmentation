import os 
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

def download_mind(data_path: str = 'mind/', mind_type: str = 'small'):
    train_news_file = os.path.join(data_path, 'train', r'news.tsv')
    valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')

    mind_url, mind_train_dataset, mind_dev_dataset, _ = get_mind_data_set(mind_type)
    
    if not os.path.exists(train_news_file):
        download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
        
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(mind_url, \
                                os.path.join(data_path, 'valid'), mind_dev_dataset)
