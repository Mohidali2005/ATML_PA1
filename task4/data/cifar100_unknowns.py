"""
this file loads the fixed near and far unknown class groups from the
official cifar hundred test partition used only for task four's final
open set evaluation
"""

import torchvision
from torch.utils.data import Dataset,DataLoader

from task4.data.cifar10 import build_transform

NEAR_CLASSES = ["bus","pickup_truck","motorcycle","tractor","wolf","fox","leopard","camel"]
FAR_CLASSES = ["bottle","bowl","chair","clock","keyboard","mushroom","sunflower","wardrobe"]

class UnknownDataset(Dataset):
    """
    this class wraps every cifar hundred test image belonging to one
    fixed group of unknown classes and applies the given transform
    """

    def __init__(self,root,class_names,transform):
        base = torchvision.datasets.CIFAR100(root=root,train=False,download=True)
        name_to_idx = {name:i for i,name in enumerate(base.classes)}
        wanted = {name_to_idx[name]:name for name in class_names}
        self.items = []
        for i in range(len(base)):
            image,label = base[i]
            if label in wanted:
                self.items.append((image,wanted[label]))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self,i):
        image,class_name = self.items[i]
        return self.transform(image),class_name

def build_unknown_loader(cfg,group):
    """
    this function builds a loader over the near or the far unknown class
    group with no augmentation
    """
    class_names = NEAR_CLASSES if group=="near" else FAR_CLASSES
    transform = build_transform(augment=False)
    dataset = UnknownDataset(cfg["paths"]["data_dir"],class_names,transform)
    return DataLoader(dataset,batch_size=cfg["data"]["eval_batch_size"],shuffle=False,num_workers=cfg["data"]["num_workers"])
