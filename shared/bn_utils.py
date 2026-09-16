"""
This file holds the batchnorm freezing policy shared by every method in
task two and task three
"""

import torch.nn as nn

def freeze_batchnorm(model):
    """
    This function switches every batchnorm layer inside the given model
    into evaluation mode so its running mean and variance stop updating
    while its scale and shift parameters stay trainable
    """
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.eval()
