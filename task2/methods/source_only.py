"""
This file trains the resnet eighteen backbone and its linear head with
ordinary cross entropy over the three labeled source domains using
domain balanced batches. No sketch image is touched during this training
and the resulting checkpoint is the erm baseline for every other method
"""

import os
import torch
import torch.nn as nn

from task2.configs.config import load_config,set_seed
from task2.methods.common import load_data,build_model,save_checkpoint
from task2.evaluation.metrics import evaluate
from task2.evaluation.history import save_training_curve
from shared.bn_utils import freeze_batchnorm
from shared.pacs_protocol import SOURCE_DOMAINS,cycle
from shared.device import DEVICE

def train_epoch(backbone,head,train_loaders,optimizer,criterion,steps_per_epoch):
    """
    This function runs one epoch by drawing one balanced batch from each
    source domain and taking one optimizer step on the combined batch
    and returns the average classification loss for the epoch
    """
    backbone.train()
    head.train()
    freeze_batchnorm(backbone)
    iterators = {domain:cycle(train_loaders[domain]) for domain in SOURCE_DOMAINS}
    total_loss = 0.0
    for _ in range(steps_per_epoch):
        images = []
        labels = []
        for domain in SOURCE_DOMAINS:
            batch_images,batch_labels,_ = next(iterators[domain])
            images.append(batch_images)
            labels.append(batch_labels)
        images = torch.cat(images,dim=0).to(DEVICE)
        labels = torch.cat(labels,dim=0).to(DEVICE)

        optimizer.zero_grad()
        features = backbone(images)
        logits = head(features)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/steps_per_epoch

def main():
    """
    This function trains the source only model watches the mean source
    validation macro f1 after every epoch stops early once it stops
    improving and saves the best backbone and head to disk
    """
    cfg = load_config("source_only")
    set_seed(cfg["seed"])

    df,splits,train_loaders,val_loaders = load_data(cfg)
    backbone,head = build_model()
    optimizer = torch.optim.AdamW(list(backbone.parameters())+list(head.parameters()),lr=cfg["optimizer"]["lr"],weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(train_loaders[domain]) for domain in SOURCE_DOMAINS)

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/source_only.pt"

    best_mean_f1 = -1
    epochs_without_improve = 0
    history = []

    for epoch in range(cfg["optimizer"]["max_epochs"]):
        cls_loss = train_epoch(backbone,head,train_loaders,optimizer,criterion,steps_per_epoch)

        val_f1s = []
        for domain in SOURCE_DOMAINS:
            accuracy,macro_f1 = evaluate(backbone,head,val_loaders[domain])
            val_f1s.append(macro_f1)
            print(f"epoch {epoch} {domain} val_acc {accuracy:.3f} val_f1 {macro_f1:.3f}")
        mean_f1 = sum(val_f1s)/len(val_f1s)
        history.append({"epoch":epoch,"cls_loss":cls_loss,"mean_val_f1":mean_f1})

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            save_checkpoint(backbone,head,checkpoint_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg["optimizer"]["patience"]:
            print(f"stopping early at epoch {epoch} best mean_f1 {best_mean_f1:.3f}")
            break

    save_training_curve(history,"source_only",cfg)

if __name__ == "__main__":
    main()
