"""
this file trains the ordinary closed set cifar resnet eighteen baseline
with plain cross entropy and standard crop and flip augmentation. every
later score and method in task four is built on top of this checkpoint
"""

import os
import torch
import torch.nn as nn

from task4.configs.config import load_config,set_seed
from task4.data.cifar10 import build_loader,load_splits,NUM_KNOWN_CLASSES
from task4.methods.common import build_model,save_checkpoint,evaluate_accuracy
from task4.evaluation.history import save_training_curve
from shared.device import DEVICE

def train_one_epoch(model,loader,optimizer,criterion):
    """
    this function runs one training epoch over the loader and returns
    the average cross entropy loss for the epoch
    """
    model.train()
    total_loss = 0.0
    for images,labels in loader:
        images,labels = images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        logits,_ = model(images)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def run_training(cfg,checkpoint_name,extra_transform=None):
    """
    this function runs a complete closed set training run for the
    configured number of epochs and keeps the checkpoint with the
    highest cifar ten validation accuracy, returning the per epoch
    history
    """
    set_seed(cfg["seed"])
    splits = load_splits(cfg)
    train_loader = build_loader(cfg,splits["train"],augment=True,batch_size=cfg["data"]["batch_size"],shuffle=True,extra_transform=extra_transform)
    val_loader = build_loader(cfg,splits["val"],augment=False,batch_size=cfg["data"]["eval_batch_size"],shuffle=False)

    model = build_model(NUM_KNOWN_CLASSES)
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.SGD(model.parameters(),lr=opt_cfg["lr"],momentum=opt_cfg["momentum"],weight_decay=opt_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt_cfg["max_epochs"])
    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt"

    best_val_acc = -1
    history = []
    for epoch in range(opt_cfg["max_epochs"]):
        train_loss = train_one_epoch(model,train_loader,optimizer,criterion)
        val_acc = evaluate_accuracy(model,val_loader)
        scheduler.step()
        print(f"epoch {epoch} loss {train_loss:.4f} val_acc {val_acc:.3f}")
        history.append({"epoch":epoch,"loss":train_loss,"val_acc":val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model,checkpoint_path)

    save_training_curve(history,checkpoint_name,cfg)
    print(f"best {checkpoint_name} val_acc {best_val_acc:.3f}")
    return history

def main():
    """
    this function runs the main vanilla training run using the settings
    from the config file
    """
    cfg = load_config("vanilla")
    run_training(cfg,"vanilla")

if __name__ == "__main__":
    main()
