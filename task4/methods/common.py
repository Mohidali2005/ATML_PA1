"""
this file has the small pieces every task four training method shares:
building a fresh model, moving it to the device, saving or loading a
checkpoint, and measuring closed set accuracy
"""

import torch

from task4.data.cifar10 import NUM_KNOWN_CLASSES
from task4.models.resnet_cifar import CifarResNet18
from shared.device import DEVICE

def build_model(num_classes):
    """
    this function builds a fresh cifar resnet eighteen sized for the
    given number of output classes and moves it onto the training device
    """
    return CifarResNet18(num_classes).to(DEVICE)

def save_checkpoint(model,path):
    """
    this function saves a model's weights to the given path
    """
    torch.save(model.state_dict(),path)

def load_checkpoint(model,path):
    """
    this function loads saved weights from the given path into a model
    """
    model.load_state_dict(torch.load(path,map_location=DEVICE))

def evaluate_accuracy(model,loader):
    """
    this function runs the model over every batch in the loader in eval
    mode and returns the top one accuracy computed from only the ten
    known class logits, so a wider proser head is scored fairly
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in loader:
            images = images.to(DEVICE)
            logits,_ = model(images)
            preds = logits[:,:NUM_KNOWN_CLASSES].argmax(dim=1).cpu()
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    return correct/total
