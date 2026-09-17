"""
This file trains the resnet eighteen backbone and its linear head with
an mmd penalty applied between every pair of the three labeled source
domains. Unlike task two's dan this never sees sketch at all so it only
tests whether invariance learned across the observed domains helps on
its own
"""

import os
import itertools
import torch
import torch.nn as nn

from task3.configs.config import load_config,set_seed
from task3.methods.common import load_data,build_model
from task3.selection.source_validation import train_with_source_selection
from task3.evaluation.history import save_training_curve
from shared.bn_utils import freeze_batchnorm
from shared.pacs_protocol import SOURCE_DOMAINS,cycle
from shared.losses import mmd_loss
from shared.device import DEVICE

DOMAIN_PAIRS = list(itertools.combinations(SOURCE_DOMAINS,2))

def make_train_epoch(backbone,head,train_loaders,optimizer,criterion,lambda_dg,steps_per_epoch):
    """
    This function returns a train epoch closure that draws one balanced
    batch from each source domain and backpropagates the classification
    loss plus the averaged pairwise mmd penalty between every source
    domain pair then reports the average classification and mmd loss for
    the epoch
    """
    def train_epoch():
        backbone.train()
        head.train()
        freeze_batchnorm(backbone)
        iterators = {domain:cycle(train_loaders[domain]) for domain in SOURCE_DOMAINS}
        total_cls_loss = 0.0
        total_mmd_loss = 0.0
        for _ in range(steps_per_epoch):
            batch_sizes = []
            images = []
            labels = []
            for domain in SOURCE_DOMAINS:
                batch_images,batch_labels,_ = next(iterators[domain])
                batch_sizes.append(batch_images.size(0))
                images.append(batch_images)
                labels.append(batch_labels)
            images = torch.cat(images,dim=0).to(DEVICE)
            labels = torch.cat(labels,dim=0).to(DEVICE)

            optimizer.zero_grad()
            features = backbone(images)
            logits = head(features)
            cls_loss = criterion(logits,labels)

            features_by_domain = dict(zip(SOURCE_DOMAINS,torch.split(features,batch_sizes,dim=0)))
            mmd = sum(mmd_loss(features_by_domain[a],features_by_domain[b]) for a,b in DOMAIN_PAIRS)/len(DOMAIN_PAIRS)

            loss = cls_loss+lambda_dg*mmd
            loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd.item()
        return {"cls_loss":total_cls_loss/steps_per_epoch,"mmd_loss":total_mmd_loss/steps_per_epoch}
    return train_epoch

def run_training(cfg,lambda_dg,checkpoint_name):
    """
    This function runs a complete dan dg training run for the given mmd
    loss weight and saves the best checkpoint under the given name
    """
    set_seed(cfg["seed"])
    df,splits,train_loaders,val_loaders = load_data(cfg)
    backbone,head = build_model()
    optimizer = torch.optim.AdamW(list(backbone.parameters())+list(head.parameters()),lr=cfg["optimizer"]["lr"],weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(train_loaders[domain]) for domain in SOURCE_DOMAINS)
    train_epoch = make_train_epoch(backbone,head,train_loaders,optimizer,criterion,lambda_dg,steps_per_epoch)

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt"
    history = train_with_source_selection(backbone,head,val_loaders,cfg,checkpoint_path,train_epoch)

    save_training_curve(history,checkpoint_name,cfg,alignment_key="mmd_loss",alignment_label="mmd loss")
    return history

def main():
    """
    This function runs the main dan dg comparison using the fixed mmd
    loss weight from the config file
    """
    cfg = load_config("dan_dg")
    run_training(cfg,cfg["dan_dg"]["lambda_dg"],"dan_dg")

if __name__ == "__main__":
    main()
