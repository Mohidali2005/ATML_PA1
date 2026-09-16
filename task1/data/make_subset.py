import json
import os
import numpy as np
from torchvision.datasets import STL10
from configs.config import load_config,set_seed

def stratified_split(labels,train_fraction,rng):
    """
    This function splits the given labels into a train part and a val part
    It works one class at a time so both parts end up class balanced
    """
    labels = np.asarray(labels)
    train_idx = []
    val_idx = []
    for c in np.unique(labels):
        idx = np.where(labels==c)[0]
        rng.shuffle(idx)
        n_train = int(round(len(idx)*train_fraction))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:].tolist())
    return sorted(train_idx),sorted(val_idx)

def balanced_subset(labels,per_class,rng):
    """
    This function picks the same number of images from every class
    If a class does not have enough images it uses all of them and
    records how many it actually used for that class
    """
    labels = np.asarray(labels)
    selected = []
    imbalance = {}
    for c in np.unique(labels):
        idx = np.where(labels==c)[0]
        rng.shuffle(idx)
        n = min(per_class,len(idx))
        if n < per_class:
            imbalance[int(c)] = n
        selected.extend(idx[:n].tolist())
    return sorted(selected),imbalance

def main():
    """
    This function downloads STL10 builds the train val split and the
    balanced evaluation subset then saves all the chosen indices to a
    json file so every later script uses the exact same images
    """
    cfg = load_config()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    train_ds = STL10(root=cfg["paths"]["stl10_root"],split="train",download=True)
    test_ds = STL10(root=cfg["paths"]["stl10_root"],split="test",download=True)

    train_idx,val_idx = stratified_split(train_ds.labels,cfg["data"]["train_val_split"],rng)
    eval_idx,imbalance = balanced_subset(test_ds.labels,cfg["data"]["eval_subset_per_class"],rng)

    os.makedirs(cfg["paths"]["cache_dir"],exist_ok=True)
    out_path = os.path.join(cfg["paths"]["cache_dir"],"splits.json")
    result = {
        "classes":train_ds.classes,
        "train_idx":train_idx,
        "val_idx":val_idx,
        "eval_idx":eval_idx,
        "eval_class_imbalance":imbalance,
    }
    with open(out_path,"w") as f:
        json.dump(result,f,indent=2)

    print(f"classes {train_ds.classes}")
    print(f"train {len(train_idx)} val {len(val_idx)} eval {len(eval_idx)}")
    if imbalance:
        print(f"some classes had fewer eval images than requested {imbalance}")
    print(f"saved splits to {out_path}")

if __name__ == "__main__":
    main()
