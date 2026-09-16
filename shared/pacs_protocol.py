"""
This file builds the source and target splits used by task two and task
three and provides the dataset class and data loaders built on top of
those splits
"""

import json
import pathlib
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T

from shared.pacs import load_pacs_dataframe,decode_image,DOMAINS,CLASSES

SOURCE_DOMAINS = ["photo","art_painting","cartoon"]
TARGET_DOMAIN = "sketch"

IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD = [0.229,0.224,0.225]

SHARED_ROOT = pathlib.Path(__file__).resolve().parent
SPLITS_PATH = SHARED_ROOT/"splits"/"pacs_sketch_seed6304.json"

def build_splits(df,seed=6304,train_fraction=0.8):
    """
    This function makes a stratified eighty twenty train and validation
    split inside each source domain and keeps every sketch row as the
    unlabeled target set
    """
    rng = np.random.default_rng(seed)
    train_idx = {}
    val_idx = {}
    for domain in SOURCE_DOMAINS:
        domain_df = df[df["domain"]==domain]
        domain_train = []
        domain_val = []
        for label in range(len(CLASSES)):
            class_idx = domain_df[domain_df["label"]==label].index.to_numpy().copy()
            rng.shuffle(class_idx)
            split_point = int(round(len(class_idx)*train_fraction))
            domain_train.extend(class_idx[:split_point].tolist())
            domain_val.extend(class_idx[split_point:].tolist())
        train_idx[domain] = domain_train
        val_idx[domain] = domain_val
    target_idx = df[df["domain"]==TARGET_DOMAIN].index.tolist()
    return {"seed":seed,"source_domains":SOURCE_DOMAINS,"target_domain":TARGET_DOMAIN,"train":train_idx,"val":val_idx,"target":target_idx}

def save_splits(splits,path=SPLITS_PATH):
    """
    This function writes the given splits dictionary to disk as json
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        json.dump(splits,f)

def load_splits(path=SPLITS_PATH):
    """
    This function reads the splits json file back into a plain python
    dictionary
    """
    with open(path) as f:
        return json.load(f)

def build_transform(train):
    """
    This function returns the image transform used for training or for
    validation and evaluation. Training adds a random crop and a random
    horizontal flip while evaluation uses a plain center crop
    """
    if train:
        return T.Compose([
            T.Resize((256,256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((256,256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN,IMAGENET_STD),
    ])

class PacsDataset(Dataset):
    """
    This dataset wraps a slice of the pacs dataframe picked out by a
    list of row indices and returns a transformed image tensor together
    with its class label and a numeric domain id
    """

    def __init__(self,df,indices,train):
        self.rows = df.loc[indices]
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,idx):
        row = self.rows.iloc[idx]
        image = decode_image(row["image"])
        image = self.transform(image)
        label = int(row["label"])
        domain = DOMAINS.index(row["domain"])
        return image,label,domain

def cycle(loader):
    """
    This generator yields batches from the given loader forever by
    starting a new pass whenever the loader runs out of batches
    """
    while True:
        for batch in loader:
            yield batch

def build_source_loaders(df,splits,batch_size_per_domain,num_workers,train):
    """
    This function builds one data loader per source domain using either
    the train or the validation indices from the given splits
    """
    split_key = "train" if train else "val"
    loaders = {}
    for domain in SOURCE_DOMAINS:
        dataset = PacsDataset(df,splits[split_key][domain],train)
        loaders[domain] = DataLoader(dataset,batch_size=batch_size_per_domain,shuffle=train,num_workers=num_workers,drop_last=train)
    return loaders

def build_target_loader(df,splits,batch_size,num_workers,train):
    """
    This function builds the single data loader over every sketch image
    used as the unlabeled adaptation set
    """
    dataset = PacsDataset(df,splits["target"],train)
    return DataLoader(dataset,batch_size=batch_size,shuffle=train,num_workers=num_workers,drop_last=train)

def main():
    """
    This function builds the pacs splits once and saves them to disk so
    every later script reuses the exact same train validation and
    target sets
    """
    df = load_pacs_dataframe()
    splits = build_splits(df)
    save_splits(splits)
    for domain in SOURCE_DOMAINS:
        print(f"{domain} train {len(splits['train'][domain])} val {len(splits['val'][domain])}")
    print(f"target {TARGET_DOMAIN} {len(splits['target'])}")

if __name__ == "__main__":
    main()
