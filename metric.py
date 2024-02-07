#!/usr/bin/python
import numpy as np

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
    # 找到label的位置
    rows, cols = np.where(where_stay_label == 1)

    # 找到label位置排第几
    ranks = np.argsort(-np.array(where_stay_pred), axis=1)
    # where_stay_pred = np.array([[1,2,3,4],[5,4,6,99],[9,8,10,6]])
    # ranks = np.array([[3, 2, 1, 0],[3, 2, 0, 1], [2, 0, 1, 3]])
    # 返回 由大到小的索引值，即先对数据生成一个0到n-1的index列表，然后按照数据从大到小排序，得到index的值

    # 组织label的位置排第几
    results = []
    for row, col in zip(rows, cols):
        results.append(np.where(ranks[row] == col)[0])
    results = np.array(results)

    return len(np.where(results < K)[0]) / len(rows)
