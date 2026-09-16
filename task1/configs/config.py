import pathlib
import random
import numpy as np
import torch
import yaml

TASK1_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = TASK1_ROOT/"configs"/"task1_config.yaml"

def load_config(path=DEFAULT_CONFIG_PATH):
    """
    This function reads the task1_config.yaml file and returns it as a
    plain python dictionary. It also turns every path listed under
    paths into a full path so scripts work no matter where they are run from
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key,rel in cfg["paths"].items():
        cfg["paths"][key] = str(TASK1_ROOT/rel)
    return cfg

def set_seed(seed):
    """
    This function fixes the random seed for the random module numpy and
    torch so every run of the code produces the same result
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
