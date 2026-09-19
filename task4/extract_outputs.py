"""
this file runs every trained model once over every evaluation split and
caches its logits and penultimate features so every score and figure
downstream reads from the same saved outputs instead of rerunning the
network
"""

import os
import torch

from task4.configs.config import load_config
from task4.data.cifar10 import build_loader,build_full_test_loader,load_splits,NUM_KNOWN_CLASSES
from task4.data.cifar100_unknowns import build_unknown_loader
from task4.methods.common import build_model,load_checkpoint
from task4.models.resnet_cifar import extend_classifier
from shared.device import DEVICE

def extract_known(model,loader):
    """
    this function runs the model over a known cifar ten loader and
    returns the stacked logits, features, and true labels
    """
    model.eval()
    all_logits,all_feats,all_labels = [],[],[]
    with torch.no_grad():
        for images,labels in loader:
            logits,feats = model(images.to(DEVICE))
            all_logits.append(logits.cpu())
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits),torch.cat(all_feats),torch.cat(all_labels)

def extract_unknown(model,loader):
    """
    this function runs the model over a cifar hundred unknown loader and
    returns the stacked logits, features, and the true cifar hundred
    class name for every image
    """
    model.eval()
    all_logits,all_feats,all_names = [],[],[]
    with torch.no_grad():
        for images,names in loader:
            logits,feats = model(images.to(DEVICE))
            all_logits.append(logits.cpu())
            all_feats.append(feats.cpu())
            all_names.extend(names)
    return torch.cat(all_logits),torch.cat(all_feats),all_names

def save_outputs(path,logits,feats,labels=None,names=None):
    """
    this function saves one cached set of model outputs to disk
    """
    torch.save({"logits":logits,"features":feats,"labels":labels,"names":names},path)

def load_cache(cfg,model_name,split):
    """
    this function loads one previously cached set of model outputs back
    from disk
    """
    path = f"{cfg['paths']['cache_dir']}/{model_name}_{split}.pt"
    return torch.load(path)

def build_model_for(cfg,model_name):
    """
    this function builds and loads the trained checkpoint for the given
    model name, widening the classifier head first when the model is
    proser
    """
    if model_name=="proser":
        model = build_model(NUM_KNOWN_CLASSES)
        extend_classifier(model,cfg["proser"]["num_dummy_classes"])
    else:
        model = build_model(NUM_KNOWN_CLASSES)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{model_name}.pt"
    load_checkpoint(model,checkpoint_path)
    return model

def extract_for_model(cfg,model_name):
    """
    this function loads one trained checkpoint and caches its outputs on
    the unaugmented cifar ten training features used to fit mahalanobis,
    the cifar ten validation and test splits, and the near and far
    unknown splits
    """
    splits = load_splits(cfg)
    model = build_model_for(cfg,model_name)

    cache_dir = cfg["paths"]["cache_dir"]
    os.makedirs(cache_dir,exist_ok=True)

    train_loader = build_loader(cfg,splits["train"],augment=False,batch_size=cfg["data"]["eval_batch_size"],shuffle=False)
    logits,feats,labels = extract_known(model,train_loader)
    save_outputs(f"{cache_dir}/{model_name}_train.pt",logits,feats,labels=labels)

    val_loader = build_loader(cfg,splits["val"],augment=False,batch_size=cfg["data"]["eval_batch_size"],shuffle=False)
    logits,feats,labels = extract_known(model,val_loader)
    save_outputs(f"{cache_dir}/{model_name}_val.pt",logits,feats,labels=labels)

    test_loader = build_full_test_loader(cfg)
    logits,feats,labels = extract_known(model,test_loader)
    save_outputs(f"{cache_dir}/{model_name}_test.pt",logits,feats,labels=labels)

    for group in ["near","far"]:
        loader = build_unknown_loader(cfg,group)
        logits,feats,names = extract_unknown(model,loader)
        save_outputs(f"{cache_dir}/{model_name}_{group}.pt",logits,feats,names=names)

    print(f"cached outputs for {model_name}")

def main():
    """
    this function extracts and caches outputs for every trained task
    four model
    """
    for method in ["vanilla","gcsc","proser"]:
        cfg = load_config(method)
        extract_for_model(cfg,method)

if __name__ == "__main__":
    main()
