"""
this file scores a proser model's example using how much probability
mass its softmax assigns to the learned dummy classifier group instead
of any of the real known classes
"""

import torch

def proser_score(full_logits,num_known):
    """
    this function turns a batch of extended proser logits into an
    unknownness score equal to the total softmax probability landing on
    the dummy classifiers
    """
    probs = torch.softmax(full_logits,dim=1)
    return probs[:,num_known:].sum(dim=1)
