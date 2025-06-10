

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)




def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def F1_score_at_k_batch(hits, k, n):
    precision = precision_at_k_batch(hits, k, n)
    recall = recall_at_k_batch(hits, k, n)

    F1 = np.zeros(len(hits), dtype=np.float32)
    for i in range(len(hits)):
        if precision[i] + recall[i] > 0:
            F1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            F1[i] = 0
    return F1


def recall_at_k_batch(hits, k, n):
    n = np.asarray(n)
    recall = np.zeros(len(hits), dtype=np.float32)
    for i in range(len(hits)):
        recall[i] = hits[i, :k].sum() / n[i]
    return recall
def precision_at_k_batch(hits, k, n):
    n = np.asarray(n)
    precision = np.zeros(len(hits), dtype=np.float32)
    for i in range(len(hits)):
        if k <= n[i]:
            precision[i] = hits[i, :k].mean()
        else:
            precision[i] = hits[i, :n[i]].sum() / n[i]
    return precision
def calc_metrics_at_k(cf_scores, train_patient_dict, test_patient_dict, patient_ids, disease_ids, Ks):
    """
    cf_scores: (n_patients, n_diseases)
    """
    test_pos_disease_binary = np.zeros([len(patient_ids), len(disease_ids)], dtype=np.float32)
    for idx, u in enumerate(patient_ids):

        test_pos_disease_list = test_patient_dict[u]
        test_pos_disease_binary[idx][test_pos_disease_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)#使用 torch.sort 对得分进行降序排序，获取排序后的索引 rank_indices
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(patient_ids)):
        binary_hit.append(test_pos_disease_binary[i][rank_indices[i]])
    binary_hit = np.array([test_pos_disease_binary[i][rank_indices[i]] for i in range(len(patient_ids))], dtype=np.float32)

    n = np.array([len(test_patient_dict[u]) for u in patient_ids])

    metrics_dict = {}

    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k, n)
        metrics_dict[k]['recall'] = recall_at_k_batch(binary_hit, k, n)
        metrics_dict[k]['F1'] = F1_score_at_k_batch(binary_hit, k, n)
        metrics_dict[k]['ndcg'] = ndcg_at_k_batch(binary_hit, k)
    return metrics_dict
