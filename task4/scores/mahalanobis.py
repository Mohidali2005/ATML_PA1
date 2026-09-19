"""
this file scores every example by its distance to the nearest known
class cluster in feature space rather than anything read off the
classifier head
"""

import numpy as np

def fit_mahalanobis(features,labels,num_classes,eps=1e-6):
    """
    this function estimates one mean feature vector per known class and
    one shared diagonal covariance from unaugmented known training
    features
    """
    means = np.stack([features[labels==c].mean(axis=0) for c in range(num_classes)])
    centered = features-means[labels]
    variance = centered.var(axis=0)+eps
    inv_variance = 1.0/variance
    return means,inv_variance

def mahalanobis_score(features,means,inv_variance):
    """
    this function returns the minimum diagonal mahalanobis distance from
    every feature to any of the fitted known class means
    """
    diffs = features[:,None,:]-means[None,:,:]
    distances = np.sum((diffs**2)*inv_variance[None,None,:],axis=2)
    return distances.min(axis=1)
