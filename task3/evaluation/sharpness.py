"""
This file measures the common sharpness proxy for every trained task
three method. It perturbs the model once along a normalized gradient
ascent direction on a fixed validation batch and reports how much the
loss increases as a standardized local diagnostic rather than a claim
about global flatness
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from task3.configs.config import load_config,set_seed
from task3.methods.common import load_data,build_model,load_checkpoint
from shared.pacs_protocol import SOURCE_DOMAINS,PacsDataset
from shared.device import DEVICE

METHODS = ["erm","dan_dg","sam"]

def build_sharpness_batch(df,splits,cfg):
    """
    This function samples a fixed number of validation images from every
    source domain and stacks them into one images tensor and one labels
    tensor shared by every method's sharpness measurement
    """
    sharp_cfg = cfg["sharpness"]
    rng = np.random.default_rng(sharp_cfg["seed"])
    indices = []
    for domain in SOURCE_DOMAINS:
        domain_val = splits["val"][domain]
        chosen = rng.choice(domain_val,size=sharp_cfg["n_per_domain"],replace=False)
        indices.extend(chosen.tolist())
    dataset = PacsDataset(df,indices,train=False)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return images.to(DEVICE),labels.to(DEVICE)

def sharpness_score(backbone,head,images,labels,cfg):
    """
    This function computes the increase in cross entropy loss on the
    given fixed batch after one normalized gradient ascent perturbation
    of the configured radius
    """
    rho = cfg["sharpness"]["rho"]
    backbone.eval()
    head.eval()
    criterion = nn.CrossEntropyLoss()
    params = list(backbone.parameters())+list(head.parameters())

    logits = head(backbone(images))
    clean_loss = criterion(logits,labels)
    grads = torch.autograd.grad(clean_loss,params)
    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]),2)
    scale = rho/(grad_norm+1e-12)
    perturbations = [g*scale for g in grads]

    with torch.no_grad():
        for p,perturbation in zip(params,perturbations):
            p.add_(perturbation)
        perturbed_logits = head(backbone(images))
        perturbed_loss = criterion(perturbed_logits,labels)
        for p,perturbation in zip(params,perturbations):
            p.sub_(perturbation)

    return (perturbed_loss-clean_loss).item()

def main():
    """
    This function computes the sharpness proxy for every trained method
    on the same fixed validation batch and saves the results to a csv
    table
    """
    cfg = load_config("erm")
    set_seed(cfg["seed"])
    df,splits,_,_ = load_data(cfg)
    images,labels = build_sharpness_batch(df,splits,cfg)

    rows = []
    for method in METHODS:
        backbone,head = build_model()
        checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{method}.pt"
        load_checkpoint(backbone,head,checkpoint_path)
        delta = sharpness_score(backbone,head,images,labels,cfg)
        print(f"{method} sharpness {delta:.4f}")
        rows.append({"method":method,"sharpness":delta})

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['tables_dir']}/sharpness.csv",index=False)

if __name__ == "__main__":
    main()
