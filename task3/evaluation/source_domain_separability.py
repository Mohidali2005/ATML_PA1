"""
This file measures how easily a linear probe can tell which of the
three source domains a trained backbone's validation features came
from. A score near chance level of one third means the representation
has become invariant across the observed source domains while a high
score means domain identity is still easy to recover
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task3.configs.config import load_config,set_seed
from task3.methods.common import load_data,build_model,load_checkpoint
from shared.pacs_protocol import SOURCE_DOMAINS
from shared.device import DEVICE

METHODS = ["erm","dan_dg","sam"]

def extract_features(backbone,loader):
    """
    This function runs the backbone over every batch in the given loader
    and stacks the resulting feature vectors into one array
    """
    backbone.eval()
    all_feats = []
    with torch.no_grad():
        for images,_,_ in loader:
            all_feats.append(backbone(images.to(DEVICE)).cpu())
    return torch.cat(all_feats,dim=0).numpy()

def collect_balanced_source_features(backbone,val_loaders,cfg):
    """
    This function extracts validation features for every source domain
    and subsamples each domain down to the smallest domain's count so
    the separability probe sees a balanced number of examples per domain
    """
    sep_cfg = cfg["source_domain_separability"]
    rng = np.random.default_rng(sep_cfg["seed"])
    feats_by_domain = {domain:extract_features(backbone,val_loaders[domain]) for domain in SOURCE_DOMAINS}
    min_count = min(len(feats) for feats in feats_by_domain.values())
    balanced = {}
    for domain,feats in feats_by_domain.items():
        indices = rng.choice(len(feats),size=min_count,replace=False)
        balanced[domain] = feats[indices]
    return balanced

def source_domain_separability_score(balanced_feats,cfg):
    """
    This function trains a multinomial logistic regression probe to tell
    the three source domains apart and returns its held out accuracy
    """
    sep_cfg = cfg["source_domain_separability"]
    features = np.concatenate([balanced_feats[domain] for domain in SOURCE_DOMAINS],axis=0)
    domain_labels = np.concatenate([np.full(len(balanced_feats[domain]),i) for i,domain in enumerate(SOURCE_DOMAINS)])
    train_feats,test_feats,train_labels,test_labels = train_test_split(features,domain_labels,test_size=sep_cfg["test_fraction"],random_state=sep_cfg["seed"],stratify=domain_labels)
    probe = LogisticRegression(C=sep_cfg["C"],max_iter=1000)
    probe.fit(train_feats,train_labels)
    return probe.score(test_feats,test_labels)

def main():
    """
    This function computes the source domain separability score for
    every trained method and saves the results to a csv table
    """
    cfg = load_config("erm")
    set_seed(cfg["seed"])
    _,_,_,val_loaders = load_data(cfg)

    rows = []
    for method in METHODS:
        backbone,head = build_model()
        checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{method}.pt"
        load_checkpoint(backbone,head,checkpoint_path)
        balanced_feats = collect_balanced_source_features(backbone,val_loaders,cfg)
        score = source_domain_separability_score(balanced_feats,cfg)
        print(f"{method} source_domain_separability {score:.3f}")
        rows.append({"method":method,"source_domain_separability":score})

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['tables_dir']}/source_domain_separability.csv",index=False)

if __name__ == "__main__":
    main()
