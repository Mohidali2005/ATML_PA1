"""
This file is a follow up stability test for cdan training, mirroring
dann_stabilized_highlr in this same folder. It reuses the exact same
training loop as cdan_stabilized, keeping gradient clipping and feature
normalization before the class conditioned domain discriminator, but
leaves the learning rate at the original config value instead of
lowering it, to check whether cdan can also stay stable at the stronger
learning rate and reach better alignment than its lowered learning rate
run
"""

from task2.configs.config import load_config
from task2.experiments.dann_stabilized import stabilized_paths
from task2.experiments.cdan_stabilized import run_training

def main():
    """
    This function runs the cdan stability test at the original learning
    rate so its result can be compared against cdan_stabilized's lowered
    learning rate run
    """
    cfg = load_config("cdan")
    cfg = stabilized_paths(cfg)
    run_training(cfg,cfg["cdan"]["max_alpha"],"cdan_stabilized_highlr")

if __name__ == "__main__":
    main()
