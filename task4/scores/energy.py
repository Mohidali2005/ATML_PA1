"""
this file scores every example using the energy summed across every
known class logit rather than only the single largest one
"""

import torch

def energy_score(logits):
    """
    this function turns a batch of known class logits into an
    unknownness score equal to the negative log sum exp energy
    """
    return -torch.logsumexp(logits,dim=1)
