import numpy as np

def check_array_type(array):
    # convert to array type 
    if not isinstance(array, (np.ndarray, np.generic)):
        array = np.array(array)
    return array

def check_array_shape(array, shape):
    # check array shape
    array = check_array_type(array)  
    if array.size == 0:
        raise ValueError("Array is empty.")
    if array.shape != shape and len(array.shape) != 1:
        try:
            array = array.reshape(shape)
        except ValueError as e:
            raise ValueError(f"Could not reshape array of shape {array.shape} to {shape}.") from e
    return array

# Reference: https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code
def precision_and_recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs):
    # referece: https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
    def calc_metrics(y_true, y_pred):
        y_true = y_true == pos_label 
        desc_sort_order = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[desc_sort_order]
        true_positives = y_true_sorted[:k].sum()
        total_positives = sum(y_true)

        precision_k = true_positives / min(k, total_positives)
        recall_k = true_positives / total_positives
        return precision_k, recall_k

    y_true = check_array_shape(y_true, (-1, 1))
    y_pred = check_array_shape(y_pred, (-1, 1))

    if y_true.shape[-1] != 1 and len(y_true.shape) > 1:
        metrics_list = [calc_metrics(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[-1])]
        precision_k_list, recall_k_list = zip(*metrics_list)
        return precision_k_list, recall_k_list
    else:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision_k, recall_k = calc_metrics(y_true, y_pred)
        return precision_k, recall_k

def precision_at_k(y_true, y_pred, k, pos_label=1, **kwargs):
    r"""Evaluation function of Precision@K"""
    precision_k, _ = precision_and_recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs)
    return precision_k

def recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs):
    r"""Evaluation function of Recall@K"""
    _, recall_k = precision_and_recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs)
    return recall_k

def ap_at_k(y_true, y_pred, k=10):
    r"""Evaluation function of AveragePrecision@K"""
    # adapted from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    y_true = check_array_shape(y_true, (-1,))
    y_pred = check_array_shape(y_pred, (-1,))
    if len(y_pred)>k:
        y_pred = y_pred[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score / min(len(y_true), k)

def map_at_k(y_true, y_pred, k=10):
    r"""Evaluation function of meanAveragePrecision@K"""
    # adapted from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    assert len(y_true.shape) == 2 and len(y_true.shape) == 2 
    assert k > 0, 'Value of k is not valid!'
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    return np.mean([ap_at_k(a,p,k) for a,p in zip(y_true, y_pred)])


def ndcg_at_k(y_true, y_pred, k=5):
    r"""
        Evaluation function of NDCG@K
    """
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert len(y_pred.shape) == 2 and len(y_pred.shape) == 2

    num_of_users, num_pos_items = y_true.shape
    sorted_ratings = -np.sort(-y_true)            # descending order !!
    discounters = np.tile([np.log2(i+1) for i in range(1, 1+num_pos_items)], (num_of_users, 1))
    normalizer_mat = (np.exp2(sorted_ratings) - 1) / discounters
    
    sort_idx = (-y_pred).argsort(axis=1)    # index of sorted predictions (max->min)
    gt_rank = np.array([np.argwhere(sort_idx == i)[:, 1]+1 for i in range(num_pos_items)]).T  # rank of the ground-truth (start from 1)
    hit = (gt_rank <= k)
    
    # calculate the normalizer first
    normalizer = np.sum(normalizer_mat[:, :k], axis=1)
    # calculate DCG
    DCG = np.sum(((np.exp2(y_true) - 1) / np.log2(gt_rank+1)) * hit.astype(float), axis=1)
    return np.mean(DCG / normalizer)