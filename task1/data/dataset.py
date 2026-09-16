"""
This file has small helper functions for loading the actual stl10
images once the train val and eval indices have already been picked
by make_subset.py
"""

import json
import numpy as np
from torchvision.datasets import STL10
from data.transforms import resize_224,to_array

def load_splits(cfg):
    """
    This function reads the splits.json file that make_subset.py wrote
    and returns it as a plain python dictionary
    """
    path = cfg["paths"]["cache_dir"]+"/splits.json"
    with open(path) as f:
        return json.load(f)

def load_stl10(cfg):
    """
    This function loads the official stl10 train and test partitions
    from disk. The data must already be downloaded by make_subset.py
    """
    train_ds = STL10(root=cfg["paths"]["stl10_root"],split="train",download=False)
    test_ds = STL10(root=cfg["paths"]["stl10_root"],split="test",download=False)
    return train_ds,test_ds

def images_to_array(dataset,indices,size=224):
    """
    This function pulls the chosen images out of a stl10 dataset
    resizes each one to two hundred twenty four pixels and stacks them
    into one array together with their labels
    """
    images = []
    labels = []
    for idx in indices:
        img,label = dataset[idx]
        img = resize_224(img,size)
        images.append(to_array(img))
        labels.append(label)
    images = np.stack(images,axis=0)
    labels = np.array(labels)
    return images,labels
