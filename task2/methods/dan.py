"""
This file trains the resnet eighteen backbone and its linear head with
the dan style mmd alignment penalty added to the ordinary source cross
entropy loss so the source and target feature distributions are pulled
together
"""

import os
import torch
import torch.nn as nn

from task2.configs.config import load_config,set_seed
from task2.methods.common import load_data,load_target_train_loader,build_model,save_checkpoint
from task2.evaluation.metrics import evaluate
from shared.bn_utils import freeze_batchnorm
from shared.pacs_protocol import SOURCE_DOMAINS,cycle
from shared.losses import mmd_loss

def train_epoch(backbone,head,train_loaders,target_iter,optimizer,criterion,lambda_mmd,steps_per_epoch):
    """
    This function runs one epoch by drawing one balanced batch from each
    source domain and one batch of unlabeled target images and taking
    one optimizer step on the classification loss plus the mmd penalty
    """
    backbone.train()
    head.train()
    freeze_batchnorm(backbone)
    iterators = {domain:cycle(train_loaders[domain]) for domain in SOURCE_DOMAINS}
    for _ in range(steps_per_epoch):
        images = []
        labels = []
        for domain in SOURCE_DOMAINS:
            batch_images,batch_labels,_ = next(iterators[domain])
            images.append(batch_images)
            labels.append(batch_labels)
        source_images = torch.cat(images,dim=0)
        source_labels = torch.cat(labels,dim=0)
        target_images,_,_ = next(target_iter)

        optimizer.zero_grad()
        source_features = backbone(source_images)
        target_features = backbone(target_images)
        logits = head(source_features)
        loss = criterion(logits,source_labels)+lambda_mmd*mmd_loss(source_features,target_features)
        loss.backward()
        optimizer.step()

def run_training(cfg,lambda_mmd,checkpoint_name):
    """
    This function runs a complete dan training run for the given mmd
    loss weight and saves the best checkpoint under the given name
    """
    set_seed(cfg["seed"])
    df,splits,train_loaders,val_loaders = load_data(cfg)
    target_loader = load_target_train_loader(df,splits,cfg)
    target_iter = cycle(target_loader)

    backbone,head = build_model()
    optimizer = torch.optim.AdamW(list(backbone.parameters())+list(head.parameters()),lr=cfg["optimizer"]["lr"],weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(train_loaders[domain]) for domain in SOURCE_DOMAINS)

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt"

    best_mean_f1 = -1
    epochs_without_improve = 0

    for epoch in range(cfg["optimizer"]["max_epochs"]):
        train_epoch(backbone,head,train_loaders,target_iter,optimizer,criterion,lambda_mmd,steps_per_epoch)

        val_f1s = []
        for domain in SOURCE_DOMAINS:
            accuracy,macro_f1 = evaluate(backbone,head,val_loaders[domain])
            val_f1s.append(macro_f1)
            print(f"epoch {epoch} {domain} val_acc {accuracy:.3f} val_f1 {macro_f1:.3f}")
        mean_f1 = sum(val_f1s)/len(val_f1s)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            save_checkpoint(backbone,head,checkpoint_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg["optimizer"]["patience"]:
            print(f"stopping early at epoch {epoch} best mean_f1 {best_mean_f1:.3f}")
            break

    return best_mean_f1

def main():
    """
    This function runs the main dan comparison using the fixed mmd loss
    weight from the config file
    """
    cfg = load_config("dan")
    run_training(cfg,cfg["dan"]["lambda_mmd"],"dan")

if __name__ == "__main__":
    main()
