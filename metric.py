#!/usr/bin/python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def naive_arg_topK(matrix, K, axis=0):
    '''
    get index of topk value, if the value is same, the previous output will be chosen
    :param matrix: negtive data
    :param K: the number be chosen
    :param axis: operation axis
    :return: index list
    '''

    full_sort = np.argsort(-np.array(matrix), axis=axis)
    return full_sort.take(np.arange(K), axis=axis)


def hitK(where_stay_pred, where_stay_label, K):

    rows, cols = np.where(where_stay_label == 1)
    ranks = np.argsort(-np.array(where_stay_pred), axis=1)

    results = []
    for row, col in zip(rows, cols):
        results.append(np.where(ranks[row] == col)[0])
    results = np.array(results)

    return len(np.where(results < K)[0]) / where_stay_label.shape[0]
    
def stay_detection_evaluation(binary_classification_label, whether_stay_pred, whether_stay_pred_prob):

    acc = accuracy_score(binary_classification_label, whether_stay_pred)
    auc = roc_auc_score(binary_classification_label, whether_stay_pred_prob)

    report = classification_report(binary_classification_label, whether_stay_pred, output_dict=True)['1.0']
    precision = report['precision']
    recall = report['recall']
    f1 = report['f1-score']
    return acc, auc, precision, recall, f1

def stay_selection_evaluation(where_stay_label, where_stay_pred):
    hit1 = hitK(where_stay_pred, where_stay_label, 1)
    hit3 = hitK(where_stay_pred, where_stay_label, 3)
    hit5 = hitK(where_stay_pred, where_stay_label, 5)
    return hit1, hit3, hit5
