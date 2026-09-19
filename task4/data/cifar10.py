"""
this file loads the official cifar ten dataset from its hugging face
mirror and builds the stratified split loaders and the full test loader
used by every task four model
"""

import json
import datasets as hf_datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
NUM_KNOWN_CLASSES = len(CLASSES)
MEAN = (0.4914,0.4822,0.4465)
STD = (0.2470,0.2435,0.2616)

def load_official_split(cache_dir,split):
    """
    this function loads one official cifar ten split from its hugging
    face mirror, downloading it into the given cache directory the first
    time it is called, since the original toronto host is often too
    slow for a colab session
    """
    return hf_datasets.load_dataset("uoft-cs/cifar10",split=split,cache_dir=cache_dir)

class CIFAR10Subset(Dataset):
    """
    this class wraps a subset of the official cifar ten training
    partition selected by index and applies the given transform
    """

    def __init__(self,cache_dir,indices,transform):
        self.base = load_official_split(cache_dir,"train")
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        row = self.base[self.indices[i]]
        return self.transform(row["img"]),row["label"]

class CIFAR10Test(Dataset):
    """
    this class wraps the complete official cifar ten test partition and
    applies the given transform
    """

    def __init__(self,cache_dir,transform):
        self.base = load_official_split(cache_dir,"test")
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self,i):
        row = self.base[i]
        return self.transform(row["img"]),row["label"]

def build_transform(augment,extra_transform=None):
    """
    this function builds the crop and flip training transform or the
    plain evaluation transform and optionally inserts one extra
    transform such as randaugment right after the crop and flip
    """
    ops = []
    if augment:
        ops.append(transforms.RandomCrop(32,padding=4))
        ops.append(transforms.RandomHorizontalFlip())
        if extra_transform is not None:
            ops.append(extra_transform)
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(MEAN,STD))
    return transforms.Compose(ops)

def load_splits(cfg):
    """
    this function reads the saved train validation split indices back
    from disk
    """
    with open(cfg["paths"]["splits_path"]) as f:
        return json.load(f)

def build_loader(cfg,indices,augment,batch_size,shuffle,extra_transform=None):
    """
    this function builds a loader over the given subset of the official
    cifar ten training partition
    """
    transform = build_transform(augment,extra_transform)
    dataset = CIFAR10Subset(cfg["paths"]["data_dir"],indices,transform)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=cfg["data"]["num_workers"])

def build_full_test_loader(cfg):
    """
    this function builds a loader over the complete official cifar ten
    test partition with no augmentation
    """
    transform = build_transform(augment=False)
    dataset = CIFAR10Test(cfg["paths"]["data_dir"],transform)
    return DataLoader(dataset,batch_size=cfg["data"]["eval_batch_size"],shuffle=False,num_workers=cfg["data"]["num_workers"])
