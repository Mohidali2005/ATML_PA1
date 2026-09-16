"""
This file measures how easily a linear probe can tell a trained
backbone's source validation features apart from its target features. A
score near fifty percent means the two domains look alike to the probe
while a high score means the representation still carries domain
identity
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task2.configs.config import load_config,set_seed
from task2.methods.common import build_model,load_checkpoint
from shared.pacs import load_pacs_dataframe
from shared.pacs_protocol import load_splits,SOURCE_DOMAINS,build_source_loaders,PacsDataset
from shared.device import DEVICE

METHODS = ["source_only","dan","dann","cdan"]

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

def collect_domain_features(backbone,df,splits,cfg,rng):
    """
    This function collects the pooled source validation features and a
    matching number of randomly sampled target features for one trained
    backbone
    """
    data_cfg = cfg["data"]
    val_loaders = build_source_loaders(df,splits,data_cfg["eval_batch_size"],data_cfg["num_workers"],train=False)
    source_feats = np.concatenate([extract_features(backbone,val_loaders[domain]) for domain in SOURCE_DOMAINS],axis=0)

    target_indices = np.array(splits["target"])
    sampled_target = rng.choice(target_indices,size=len(source_feats),replace=False).tolist()
    target_dataset = PacsDataset(df,sampled_target,train=False)
    target_loader = DataLoader(target_dataset,batch_size=data_cfg["eval_batch_size"],shuffle=False,num_workers=data_cfg["num_workers"])
    target_feats = extract_features(backbone,target_loader)

    return source_feats,target_feats

def domain_separability_score(source_feats,target_feats,cfg):
    """
    This function trains a logistic regression probe to separate source
    features from target features and returns its held out accuracy
    """
    sep_cfg = cfg["domain_separability"]
    features = np.concatenate([source_feats,target_feats],axis=0)
    labels = np.concatenate([np.zeros(len(source_feats)),np.ones(len(target_feats))])
    train_feats,test_feats,train_labels,test_labels = train_test_split(features,labels,test_size=sep_cfg["test_fraction"],random_state=sep_cfg["seed"],stratify=labels)
    probe = LogisticRegression(C=sep_cfg["C"],max_iter=1000)
    probe.fit(train_feats,train_labels)
    return probe.score(test_feats,test_labels)

def main():
    """
    This function computes the domain separability score for every
    trained method and saves the results to a csv table
    """
    cfg = load_config("source_only")
    set_seed(cfg["seed"])
    df = load_pacs_dataframe()
    splits = load_splits()
    rng = np.random.default_rng(cfg["domain_separability"]["seed"])

    rows = []
    for method in METHODS:
        backbone,head = build_model()
        checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{method}.pt"
        load_checkpoint(backbone,head,checkpoint_path)
        source_feats,target_feats = collect_domain_features(backbone,df,splits,cfg,rng)
        score = domain_separability_score(source_feats,target_feats,cfg)
        print(f"{method} domain_separability {score:.3f}")
        rows.append({"method":method,"domain_separability":score})

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['tables_dir']}/domain_separability.csv",index=False)

if __name__ == "__main__":
    main()
