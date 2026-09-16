"""
This file has the data loading and checkpoint helpers shared by every
training method in task two
"""

import torch

from shared.pacs import load_pacs_dataframe,CLASSES
from shared.pacs_protocol import load_splits,build_source_loaders,build_target_loader
from shared.device import DEVICE
from task2.models.backbone import ResnetBackbone
from task2.models.classifier_head import ClassifierHead

def load_data(cfg):
    """
    This function loads the pacs dataframe and the saved splits and
    builds the source training loaders and the source validation loaders
    used by every method
    """
    df = load_pacs_dataframe()
    splits = load_splits()
    data_cfg = cfg["data"]
    train_loaders = build_source_loaders(df,splits,data_cfg["batch_size_per_domain"],data_cfg["num_workers"],train=True)
    val_loaders = build_source_loaders(df,splits,data_cfg["eval_batch_size"],data_cfg["num_workers"],train=False)
    return df,splits,train_loaders,val_loaders

def load_target_train_loader(df,splits,cfg):
    """
    This function builds the loader over the unlabeled sketch images
    used by the three adaptation methods during training
    """
    data_cfg = cfg["data"]
    return build_target_loader(df,splits,data_cfg["target_batch_size"],data_cfg["num_workers"],train=True)

def load_target_eval_loader(df,splits,cfg):
    """
    This function builds the loader over every sketch image used to
    score a trained model at the end of a run
    """
    data_cfg = cfg["data"]
    return build_target_loader(df,splits,data_cfg["eval_batch_size"],data_cfg["num_workers"],train=False)

def build_model():
    """
    This function builds a fresh resnet backbone and a fresh linear head
    sized for the pacs classes and moves both onto the training device
    """
    backbone = ResnetBackbone().to(DEVICE)
    head = ClassifierHead(backbone.feature_dim,len(CLASSES)).to(DEVICE)
    return backbone,head

def save_checkpoint(backbone,head,path):
    """
    This function saves the backbone and head weights together in one
    checkpoint file
    """
    torch.save({"backbone":backbone.state_dict(),"head":head.state_dict()},path)

def load_checkpoint(backbone,head,path):
    """
    This function loads a saved backbone and head checkpoint back into
    the given modules
    """
    checkpoint = torch.load(path,map_location=DEVICE)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
