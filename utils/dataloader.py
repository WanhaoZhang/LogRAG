
import re
import pickle
import string
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import time
from datetime import datetime
import csv 


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

def load_log_tokens(dataset_name, log_file, train_ratio=1.0):
    print("Loading", log_file)

    with open(log_file, mode="r", encoding='utf8') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]
    E = {}

    print("Loaded", len(logs), "lines!")
    i = 0
    failure_count = 0
    n_train = int(len(logs) * train_ratio)
    c = 0
    contents = []
    labels   = []
    lineids  = []
    while i < n_train:
        c += 1
        if c % 1000 == 0:
            print("\rLoading {0:.2f}% - {1} unique logs".format(i * 100 / n_train, len(E.keys())), end="")
        if logs[i][0] != "-":
            failure_count += 1
        
        label = logs[i][0]
        content = logs[i]
        content = content[content.find(' ') + 1:]
        content = clean(content.lower())
        
        contents.append(content)
        labels.append(label)
        lineids.append(i)
        
        i += 1
       
    print("\nlast train index:", i)
    ###
    # 将两个列表合并为一个元组的列表
    data = list(zip(lineids, contents, labels))
    # 指定保存的文件名
    csv_file_path = "BGL_log_tokens.csv"
    # 使用 csv 模块写入 CSV 文件
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['LineId', 'Content', 'Label'])
        writer.writerows(data)

    return csv_file_path
