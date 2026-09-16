"""
This file holds the maximum mean discrepancy loss used by dan in task
two and reused unchanged by dan dg in task three
"""

import torch

def pairwise_squared_distance(x,y):
    """
    This function returns the matrix of squared euclidean distances
    between every row of x and every row of y
    """
    x_sq = (x**2).sum(dim=1,keepdim=True)
    y_sq = (y**2).sum(dim=1,keepdim=True)
    return x_sq+y_sq.t()-2*x@y.t()

def rbf_kernel_sum(x,y,median_dist,bandwidth_multipliers=(0.5,1.0,2.0)):
    """
    This function evaluates a sum of three rbf kernels between every row
    of x and every row of y using the given median pairwise squared
    distance to set each kernel bandwidth
    """
    dist = pairwise_squared_distance(x,y)
    kernel = 0.0
    for multiplier in bandwidth_multipliers:
        bandwidth = multiplier*median_dist
        kernel = kernel+torch.exp(-dist/bandwidth)
    return kernel

def mmd_loss(source_feats,target_feats,bandwidth_multipliers=(0.5,1.0,2.0)):
    """
    This function computes the squared maximum mean discrepancy between
    a batch of source features and a batch of target features. It sums
    three rbf kernels whose bandwidths are the given multipliers times
    the median pairwise squared distance across the combined batch
    """
    combined = torch.cat([source_feats,target_feats],dim=0)
    median_dist = pairwise_squared_distance(combined,combined).median().clamp(min=1e-8)
    k_ss = rbf_kernel_sum(source_feats,source_feats,median_dist,bandwidth_multipliers).mean()
    k_tt = rbf_kernel_sum(target_feats,target_feats,median_dist,bandwidth_multipliers).mean()
    k_st = rbf_kernel_sum(source_feats,target_feats,median_dist,bandwidth_multipliers).mean()
    return k_ss+k_tt-2*k_st
