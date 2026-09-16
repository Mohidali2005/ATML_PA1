"""
This file trains one linear classifier head for every backbone using
its cached frozen features. Training stops early once the validation
accuracy has not improved for a set number of epochs and the head
from the best epoch is the one that gets saved
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from configs.config import load_config,set_seed

def train_one_head(train_feats,train_labels,val_feats,val_labels,cfg):
    """
    This function trains a single linear layer on the given frozen
    features. It watches the validation accuracy after every epoch and
    keeps a copy of the weights from whichever epoch was best so far
    """
    head_cfg = cfg["classifier_head"]
    num_classes = int(train_labels.max().item())+1
    head = nn.Linear(train_feats.shape[1],num_classes)
    optimizer = torch.optim.AdamW(head.parameters(),lr=head_cfg["lr"],weight_decay=head_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    dataset = data.TensorDataset(train_feats,train_labels)
    loader = data.DataLoader(dataset,batch_size=head_cfg["batch_size"],shuffle=True)

    best_val_acc = -1
    best_state = None
    epochs_without_improve = 0

    for epoch in range(head_cfg["max_epochs"]):
        head.train()
        for batch_feats,batch_labels in loader:
            optimizer.zero_grad()
            outputs = head(batch_feats)
            loss = criterion(outputs,batch_labels)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_outputs = head(val_feats)
            val_acc = (val_outputs.argmax(1)==val_labels).float().mean().item()

        print(f"epoch {epoch} val_acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key:value.clone() for key,value in head.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= head_cfg["patience"]:
            print(f"stopping early at epoch {epoch} best val_acc {best_val_acc:.3f}")
            break

    head.load_state_dict(best_state)
    return head,best_val_acc

def main():
    """
    This function loads the cached train and val features for every
    backbone trains a head on each one and saves the trained weights
    """
    cfg = load_config()
    set_seed(cfg["seed"])

    features_dir = cfg["paths"]["cache_dir"]+"/features"
    labels = np.load(features_dir+"/labels.npz")
    train_labels = torch.from_numpy(labels["train"]).long()
    val_labels = torch.from_numpy(labels["val"]).long()

    heads_dir = cfg["paths"]["cache_dir"]+"/heads"
    os.makedirs(heads_dir,exist_ok=True)

    for backbone_name in ["resnet","vit","clip"]:
        train_feats = torch.load(f"{features_dir}/{backbone_name}_train.pt")
        val_feats = torch.load(f"{features_dir}/{backbone_name}_val.pt")

        print(f"training head for {backbone_name}")
        head,best_val_acc = train_one_head(train_feats,train_labels,val_feats,val_labels,cfg)

        torch.save(head.state_dict(),f"{heads_dir}/{backbone_name}_head.pt")
        print(f"{backbone_name} best val_acc {best_val_acc:.3f}")

if __name__ == "__main__":
    main()
