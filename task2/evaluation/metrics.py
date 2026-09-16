"""
This file has the evaluation helpers shared by every training script and
by the final comparison tables
"""

import torch
from sklearn.metrics import f1_score

def predict(backbone,head,loader):
    """
    This function runs the backbone and the head over every batch in the
    given loader and returns the predicted labels together with the true
    labels for the whole loader
    """
    backbone.eval()
    head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images,labels,_ in loader:
            features = backbone(images)
            logits = head(features)
            all_preds.append(logits.argmax(dim=1))
            all_labels.append(labels)
    return torch.cat(all_preds),torch.cat(all_labels)

def evaluate(backbone,head,loader):
    """
    This function returns the accuracy and the macro f1 score of the
    backbone and head on the given loader
    """
    preds,labels = predict(backbone,head,loader)
    accuracy = (preds==labels).float().mean().item()
    macro_f1 = f1_score(labels.numpy(),preds.numpy(),average="macro")
    return accuracy,macro_f1

def per_class_accuracy(backbone,head,loader,num_classes):
    """
    This function returns the accuracy for every individual class on the
    given loader as a plain dictionary keyed by the numeric class label
    """
    preds,labels = predict(backbone,head,loader)
    accuracies = {}
    for label in range(num_classes):
        mask = labels==label
        accuracies[label] = (preds[mask]==labels[mask]).float().mean().item()
    return accuracies
