"""
this file builds the stratified ninety ten split of the official cifar
ten training partition used by every task four training run and saves
the resulting indices so the same split is reused everywhere
"""

import json
import os
import torchvision
from sklearn.model_selection import train_test_split

from task4.configs.config import load_config

def main():
    """
    this function downloads the official cifar ten training partition if
    needed and saves a stratified ninety ten train validation split to a
    json file keyed by seed six three zero four
    """
    cfg = load_config("vanilla")
    dataset = torchvision.datasets.CIFAR10(root=cfg["paths"]["data_dir"],train=True,download=True)
    labels = dataset.targets

    train_idx,val_idx = train_test_split(
        range(len(labels)),
        test_size=cfg["data"]["val_fraction"],
        random_state=cfg["seed"],
        stratify=labels,
    )

    splits_path = cfg["paths"]["splits_path"]
    os.makedirs(os.path.dirname(splits_path),exist_ok=True)
    with open(splits_path,"w") as f:
        json.dump({"train":sorted(train_idx),"val":sorted(val_idx)},f)
    print(f"saved {len(train_idx)} train and {len(val_idx)} val indices to {splits_path}")

if __name__ == "__main__":
    main()
