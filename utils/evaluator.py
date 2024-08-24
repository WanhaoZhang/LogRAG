from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    return tp, fp, tn, fn, acc, pre, rec, f1


def time_sliding_window(y_true,y_pred, time_data, window_size, step_size):

    log_size = len(time_data)
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + window_size
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    while end_index < log_size:
        start_time = start_time + step_size
        end_time = start_time + window_size
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

    y_pred_windowed = []
    y_true_windowed = []
    for (start_index, end_index) in start_end_index_pair:
        if any(y_pred[start_index:end_index]):
            y_pred_windowed.append(1)
        else:
            y_pred_windowed.append(0)
            
        if any(y_true[start_index:end_index]):
            y_true_windowed.append(1)
        else:
            y_true_windowed.append(0)

    return y_true_windowed, y_pred_windowed



def evaluate(configs, test_data_path, anomaly_lineid_list, logger):
    
    dataset_name = configs['dataset_name']
    window_size = configs['window_size']
    seconds = configs['window_time']

    
    df_test = pd.read_csv(test_data_path)
    # get the acc pre rec f1 scores
    idxs = df_test['LineId'].tolist()
    
    if dataset_name == "HDFS":
        y_true = df_test['Label'].tolist()
    else :
        y_true = df_test['Label'].apply(lambda x : 0 if x == '-' else 1).tolist()
        
    y_true = np.array(y_true)
    
    y_pred_dict = {idx: 0 for idx in idxs}
    for x in anomaly_lineid_list:
        if x in y_pred_dict:
            y_pred_dict[x] = 1
            
    assert set(y_pred_dict.keys()) == set(idxs), "y_pre_dict do not have same idxs in prediction"
    
    y_pred = np.array([y_pred_dict[idx] for idx in idxs])
    
    
    TP, FP, TN, FN, acc, pre, rec, f1 = calculate_metrics(y_true, y_pred)
    logger.info("fixing windows size 1:  TP: {}, FP: {}, TN: {}, FN: {} ".format(TP, FP, TN, FN))
    logger.info(f"fixing windows size {1}: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}\n")
    
   
    y_pred_windowed = [1 if any(y_pred[i:i + window_size]) else 0 for i in range(0, len(y_pred), window_size)]
    y_true_windowed = [1 if any(y_true[i:i + window_size]) else 0 for i in range(0, len(y_true), window_size)]
    wdtp, wdfp, wdtn, wdfn, wdacc, wdpre, wdrec, wdf1 = calculate_metrics(y_true_windowed, y_pred_windowed)
    logger.info(f"fixing windows size {window_size}, Tp: {wdtp}, Fp: {wdfp}, Tn: {wdtn}, Fn: {wdfn}")
    logger.info(f"fixing windows size {window_size}, Acc: {wdacc:.4f}, Precision: {wdpre:.4f}, Recall: {wdrec:.4f}, F1: {wdf1:.4f}\n")    

    y_true_windowed, y_pred_windowed = time_sliding_window(y_true, y_pred, df_test['Timestamp'].tolist(), window_size=seconds, step_size=seconds)
    wdtp, wdfp, wdtn, wdfn, wdacc, wdpre, wdrec, wdf1 = calculate_metrics(y_true_windowed, y_pred_windowed)
    logger.info(f"time windows size {seconds} seconds, Tp: {wdtp}, Fp: {wdfp}, Tn: {wdtn}, Fn: {wdfn}")
    logger.info(f"time windows size {seconds} seconds, Acc: {wdacc:.4f}, Precision: {wdpre:.4f}, Recall: {wdrec:.4f}, F1: {wdf1:.4f}\n")     