"""
this file builds the manifold mixup data placeholders proser trains
toward its dummy classifiers. it blends the layer two feature map of two
training examples from different classes rather than blending raw
pixels
"""

import torch

def sample_mixup_partners(labels):
    """
    this function returns one partner index per example that never
    shares that example's class label, starting from a random
    permutation and resampling only the positions where the permutation
    happened to pair an example with its own class
    """
    n = labels.size(0)
    partner = torch.randperm(n,device=labels.device)
    same_class = labels[partner]==labels
    while same_class.any():
        partner[same_class] = torch.randint(0,n,(int(same_class.sum()),),device=labels.device)
        same_class = labels[partner]==labels
    return partner

def manifold_mixup(model,images,labels,alpha):
    """
    this function runs the batch through the network up to layer two,
    blends each example's feature map with a differently labeled
    partner's feature map using a beta sampled mixing weight, and
    returns the resulting mixed feature map ready to continue through
    the rest of the network
    """
    partner = sample_mixup_partners(labels)
    pre_features = model.forward_pre(images)
    lam = torch.distributions.Beta(alpha,alpha).sample().item()
    mixed = lam*pre_features+(1-lam)*pre_features[partner]
    return mixed
