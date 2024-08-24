import numpy as np
import pandas as pd
import os
import logging
from prelogad.DeepSVDD.src.deepSVDD import DeepSVDD
from prelogad.DeepSVDD.src.datasets.main import load_dataset
from tqdm import tqdm 
import logging
from postprocess import RAG
import yaml
from utils.evaluator import evaluate

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)
    
api_key = configs['api_key']
os.environ["OPENAI_API_BASE"] = configs['api_base']
os.environ["OPENAI_API_KEY"] = api_key
# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./output/runtime.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def train_deepsvdd(train_data_path):
    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    deep_SVDD = DeepSVDD('soft-boundary')
    deep_SVDD.set_network("mlp")
    
    if not configs['is_train']:
        deep_SVDD.load_model(model_path='./output/model.tar', load_ae=False)
        logger.info('Loading model from ./output/model.tar' )
    else :
        # dataloader 
        train_dataset = load_dataset(data_path=train_data_path, encoder_path=configs['encoder_path'])
        # pretrain and train 
        if configs['is_pretrain']:
            deep_SVDD.pretrain( train_dataset,
                                optimizer_name=configs['optimizer_name'],
                                lr=configs['lr'],
                                n_epochs=configs['n_epochs'],
                                lr_milestones=configs['lr_milestones'],
                                batch_size=configs['batch_size'],
                                weight_decay=configs['weight_decay'],
                                device=configs['device'],
                                n_jobs_dataloader=configs['n_jobs_dataloader'])
        deep_SVDD.train(train_dataset,
                        optimizer_name=configs['optimizer_name'],  
                        lr=configs['lr'],  
                        n_epochs=configs['n_epochs'],  
                        lr_milestones=configs['lr_milestones'],  
                        batch_size=configs['batch_size'],  
                        weight_decay=configs['weight_decay'],  
                        device=configs['device'],  
                        n_jobs_dataloader=configs['n_jobs_dataloader'])  
    # Save results, model, and configuration
    model_path = './output/model.tar'
    deep_SVDD.save_results(export_json='./output/results.json')
    deep_SVDD.save_model(export_model= './output/model.tar', save_ae=False)
    return model_path

def anomaly_detection(model_path, test_data_path):
    logger.info("start testing....")
    
    deep_SVDD = DeepSVDD('soft-boundary')
    deep_SVDD.set_network("mlp")
    deep_SVDD.load_model(model_path=model_path, load_ae=False)
    logger.info('Loading model from ./output/model.tar' )
    
    test_dataset = load_dataset(data_path=test_data_path, encoder_path=configs['encoder_path'])
    anomalys, _ = deep_SVDD.test(test_dataset, device='cpu', n_jobs_dataloader=configs['n_jobs_dataloader'])   
    # (idx, label, score)
    anomaly_lineid_list = []
    for item in tqdm(anomalys, desc='saving anomaly LineIds to list'):
        idx = item[0]
        lineid = idx 
        anomaly_lineid_list.append(lineid)
        
    output_file = 'output/anomaly_logs_detc_by_svdd.csv'
    # 保存deepsvdd检测为异常的
    df_test = pd.read_csv(test_data_path)
    pos_index = []
    for item in tqdm(anomalys, desc='saving anomalys detected by svdd to csv'):
        idx = item[0]
        pos_index.append(idx)
    pos_index.sort()
    pos_df = df_test[df_test["LineId"].isin(pos_index)]
    pos_df.to_csv(output_file, index=False)
    
    return output_file, anomaly_lineid_list


def main():
    all_df = pd.read_csv(configs['log_structed_path'])
    num_train = int(configs['train_ratio']*len(all_df))

    train_df = all_df[:num_train]
    train_df = train_df[train_df['Label'] == '-']
    test_df = all_df[num_train:]
    
    
    train_log_structed_path = f"./dataset/{configs['dataset_name']}/train_log_structured.csv"
    test_log_structed_path = f"./dataset/{configs['dataset_name']}/test_log_structured.csv"

    train_df.to_csv(train_log_structed_path, index=False)
    test_df.to_csv(test_log_structed_path, index=False)
    
    # train deepsvdd, get log token embeddings
    model_path = train_deepsvdd(train_log_structed_path)
    # do anomaly detection
    anomaly_logs_path, anomaly_lineid_list = anomaly_detection(model_path, test_log_structed_path)
    
    # rag postporcessing, get log templates embeddings
    if configs['is_rag']:
        RagPoster = RAG.RAGPostProcessor(configs, train_data_path=train_log_structed_path)
        anomaly_lineid_list = RagPoster.post_process(anomaly_logs_path, test_log_structed_path)
    # print final results
    evaluate(configs, test_log_structed_path, anomaly_lineid_list, logger)
    

if __name__ == '__main__':
    main()