"""
This file wraps the resnet eighteen backbone used by every method in
task three. Every method here fine tunes the whole backbone rather than
freezing it whether that method is ordinary erm dan dg alignment or sam
"""

import torch.nn as nn
import torchvision

class ResnetBackbone(nn.Module):
    """
    This class wraps torchvision resnet eighteen pretrained on imagenet
    and exposes its five hundred twelve dimensional pooled feature
    instead of the thousand class scores it was originally trained with
    """

    feature_dim = 512

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Identity()
        self.model = model

    def forward(self,images):
        """
        This method expects a batch of already normalized images and
        returns the pooled feature for every image in the batch
        """
        return self.model(images)
