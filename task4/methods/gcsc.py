"""
this file trains a second ten class closed set baseline that adds
randaugment on top of the ordinary crop and flip pipeline used by the
vanilla model, so the evaluation can test whether stronger augmentation
alone changes rejection quality
"""

import torchvision.transforms as transforms

from task4.configs.config import load_config
from task4.methods.vanilla import run_training

def main():
    """
    this function runs the gcsc training run using the vanilla training
    loop with a randaugment transform inserted right after crop and flip
    """
    cfg = load_config("gcsc")
    gcsc_cfg = cfg["gcsc"]
    extra_transform = transforms.RandAugment(num_ops=gcsc_cfg["randaugment_num_ops"],magnitude=gcsc_cfg["randaugment_magnitude"])
    run_training(cfg,"gcsc",extra_transform=extra_transform)

if __name__ == "__main__":
    main()
