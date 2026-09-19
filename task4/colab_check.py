"""
this file is a quick colab sanity check run once before the real
training stages. it confirms a gpu is visible and that the near and far
unknown class names line up with cifar hundred's own fine label names
before any long training run starts
"""

import torch

from task4.configs.config import load_config
from task4.data.cifar100_unknowns import build_unknown_loader

def main():
    """
    this function prints whether cuda is available and reports the near
    and far unknown loader sizes so a class name mismatch shows up
    immediately instead of after a long training run
    """
    print("cuda available",torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device name",torch.cuda.get_device_name(0))

    cfg = load_config("vanilla")
    near_loader = build_unknown_loader(cfg,"near")
    far_loader = build_unknown_loader(cfg,"far")
    print("near unknown count",len(near_loader.dataset))
    print("far unknown count",len(far_loader.dataset))

if __name__ == "__main__":
    main()
