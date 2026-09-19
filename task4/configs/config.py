"""
this file loads the base config and merges a chosen method config on
top of it so every task four script gets one plain dictionary of
settings
"""

import pathlib
import random
import numpy as np
import torch
import yaml

TASK4_ROOT = pathlib.Path(__file__).resolve().parents[1]
PA1_ROOT = TASK4_ROOT.parent

def load_config(method):
    """
    this function reads the base config and the config file for the
    given method merges the method settings on top of the base settings
    and turns every path listed under paths into a full path so scripts
    work no matter where they are run from
    """
    with open(TASK4_ROOT/"configs"/"base.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(TASK4_ROOT/"configs"/f"{method}.yaml") as f:
        method_cfg = yaml.safe_load(f)
    if method_cfg:
        cfg.update(method_cfg)
    for key,rel in cfg["paths"].items():
        cfg["paths"][key] = str(PA1_ROOT/rel)
    return cfg

def set_seed(seed):
    """
    this function fixes the random seed for the random module numpy and
    torch so every run of the code produces the same result
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
