"""
this file scores every example by how much probability mass its
predicted class is missing, using only the closed set logits
"""

import torch

def msp_score(logits):
    """
    this function turns a batch of known class logits into an
    unknownness score equal to one minus the maximum softmax probability
    """
    probs = torch.softmax(logits,dim=1)
    return 1-probs.max(dim=1).values
