
from prelogad.DeepSVDD.src.base.torchvision_dataset import TorchvisionDataset
from prelogad.DeepSVDD.src.datasets.vocab import Vocab
import pandas as pd 
import torch
import json
from tqdm import tqdm
from tqdm import tqdm
# import tensorflow as tf
from transformers import AutoTokenizer
from transformers import BertModel
import torch
import re
import pickle
import string

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

# version 2 use log tokens

def bert_encoder(s,  biglog_tokenizer, pretrained_model, no_wordpiece=0):
    """ Compute semantic vector with BERT
    Parameters
    ----------
    s: string to encode
    no_wordpiece: 1 if you do not use sub-word tokenization, otherwise 0

    Returns
    -------
        np array in shape of (768,)
    """
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in biglog_tokenizer.vocab.keys()]
        s = " ".join(words)
    # inputs = biglog_tokenizer(s, return_tensors='tf', max_length=512)
    # outputs = pretrained_model(**inputs)
    tokenized_data=biglog_tokenizer(s, padding = "longest", truncation=True, max_length=150)
    outputs=pretrained_model(torch.tensor(tokenized_data['input_ids']).unsqueeze(0))
    # v = tf.reduce_mean(outputs.last_hidden_state, 1)
    v = torch.mean(outputs.last_hidden_state, dim=1)
    # print(v[0].shape)
    return v[0]

def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message

    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s



class BGL_Dataset(TorchvisionDataset):
    def __init__(self, root: str, encoder_path: str):
        '''
            root: data_path
        '''
        biglog_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        pretrained_model = BertModel.from_pretrained(encoder_path)


        df_logs = pd.read_csv(root)
        train_set = []
        test_set = []
        # 遍历DataFrame中的每一行
        E = {}
        for _, row in tqdm(df_logs.iterrows(), total=len(df_logs)):
            lineid  = row['LineId']
            content = row['Content']

            if "HDFS" in root:
                label = row['Label']
            else:
                label   = int(row['Label'] != '-')
            
            try:
                content = content[content.find(' ') + 1:]
                content = clean(content.lower())
            except:
                continue
            
            if content not in E.keys():
                E[content] = bert_encoder(content,biglog_tokenizer, pretrained_model, no_wordpiece=0)
            log_embedding = E[content]

            if label == 0:
                train_set.append([log_embedding.clone().detach().float(), 0, lineid])
            
            test_set.append([log_embedding.clone().detach().float(), label, lineid])
            
        self.train_set = train_set
        self.test_set  = test_set

