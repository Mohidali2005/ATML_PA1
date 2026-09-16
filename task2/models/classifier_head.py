"""
This file holds the small linear layer that turns a resnet feature into
class scores for the seven pacs categories
"""

import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    This class is a single linear layer mapping a resnet feature to
    class scores
    """

    def __init__(self,feature_dim,num_classes):
        super().__init__()
        self.linear = nn.Linear(feature_dim,num_classes)

    def forward(self,features):
        """
        This method maps a batch of features to a batch of class scores
        """
        return self.linear(features)
