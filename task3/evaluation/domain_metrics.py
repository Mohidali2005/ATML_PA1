"""
This file has the evaluation helpers shared by every task three script.
It is also the only place that knows how to build the sketch loader so
that loader only ever gets built from the final evaluation scripts
"""

import torch
from sklearn.metrics import f1_score

from shared.pacs_protocol import build_target_loader
from shared.device import DEVICE

def predict(backbone,head,loader):
    """
    This function runs the backbone and the head over every batch in the
    given loader and returns the predicted labels together with the true
    labels for the whole loader on the cpu
    """
    backbone.eval()
    head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images,labels,_ in loader:
            features = backbone(images.to(DEVICE))
            logits = head(features)
            all_preds.append(logits.argmax(dim=1).cpu())
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

def mean_and_worst(values):
    """
    This function returns the mean and the minimum of a list of per
    domain values
    """
    return sum(values)/len(values),min(values)

def load_target_eval_loader(df,splits,cfg):
    """
    This function builds the loader over every sketch image. It is only
    called from the final evaluation scripts after every task three
    checkpoint and setting has already been fixed
    """
    data_cfg = cfg["data"]
    return build_target_loader(df,splits,data_cfg["eval_batch_size"],data_cfg["num_workers"],train=False)
