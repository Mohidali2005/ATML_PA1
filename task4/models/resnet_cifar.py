"""
this file builds the cifar sized resnet eighteen backbone shared by
every method in task four
"""

import torch
import torch.nn as nn
import torchvision

class CifarResNet18(nn.Module):
    """
    this class is a torchvision resnet eighteen with its imagenet stem
    replaced by a stride one three by three convolution and no initial
    max pooling so it keeps the full spatial resolution of a small
    thirty two by thirty two image, and it exposes both the classifier
    logits and the penultimate pooled feature
    """

    feature_dim = 512

    def __init__(self,num_classes):
        super().__init__()
        model = torchvision.models.resnet18(weights=None,num_classes=num_classes)
        model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        model.maxpool = nn.Identity()
        self.stem = nn.Sequential(model.conv1,model.bn1,model.relu)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward_pre(self,images):
        """
        this method runs the stem through layer two and returns the
        intermediate feature map that manifold mixup blends between two
        examples
        """
        h = self.stem(images)
        h = self.layer1(h)
        h = self.layer2(h)
        return h

    def forward_post(self,h):
        """
        this method runs layer three through the classifier head on a
        feature map and returns the logits together with the pooled
        penultimate feature
        """
        h = self.layer3(h)
        h = self.layer4(h)
        pooled = self.avgpool(h).flatten(1)
        logits = self.fc(pooled)
        return logits,pooled

    def forward(self,images):
        """
        this method runs a batch of images through the whole network and
        returns the logits together with the pooled penultimate feature
        """
        h = self.forward_pre(images)
        return self.forward_post(h)

def extend_classifier(model,num_extra):
    """
    this function replaces a model's final linear layer with a wider one
    that keeps the original class weights in the first rows and appends
    num_extra freshly initialized rows for the proser dummy classifiers,
    building the new layer directly on the old one's device so the
    weight copy never crosses devices
    """
    old_fc = model.fc
    device = old_fc.weight.device
    new_fc = nn.Linear(old_fc.in_features,old_fc.out_features+num_extra).to(device)
    with torch.no_grad():
        new_fc.weight[:old_fc.out_features] = old_fc.weight
        new_fc.bias[:old_fc.out_features] = old_fc.bias
    model.fc = new_fc
    return model
