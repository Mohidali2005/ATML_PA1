"""
This file is a follow up stability test for dann training. It reuses the
exact same training loop as dann_stabilized in this same folder, keeping
gradient clipping and feature normalization before the domain
discriminator, but leaves the learning rate at the original config value
instead of lowering it. The goal is to isolate whether the learning rate
drop was the part of dann_stabilized that actually fixed the collapse, or
whether clipping and normalization alone are enough to keep training
stable at the stronger original learning rate, which would let the
adversarial signal push the backbone harder and hopefully produce real
domain alignment instead of just a stable but inert classifier
"""

from task2.configs.config import load_config
from task2.experiments.dann_stabilized import stabilized_paths,run_training

def main():
    """
    This function runs the dann stability test at the original learning
    rate so its result can be compared against dann_stabilized's lowered
    learning rate run
    """
    cfg = load_config("dann")
    cfg = stabilized_paths(cfg)
    run_training(cfg,cfg["dann"]["max_alpha"],"dann_stabilized_highlr")

if __name__ == "__main__":
    main()
