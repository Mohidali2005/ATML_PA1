"""
this file has the open set recognition metrics shared by every score
and method comparison in task four
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def auroc(known_scores,unknown_scores):
    """
    this function returns the area under the roc curve for separating
    the given unknown scores from the given known scores, where a larger
    score means more novel
    """
    scores = np.concatenate([known_scores,unknown_scores])
    labels = np.concatenate([np.zeros(len(known_scores)),np.ones(len(unknown_scores))])
    return roc_auc_score(labels,scores)

def acceptance_rate(scores,threshold):
    """
    this function returns the fraction of the given scores at or below
    the threshold, meaning the example is accepted as a known class
    """
    return float(np.mean(scores<=threshold))
