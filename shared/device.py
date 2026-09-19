"""
This file picks the device every task script trains on so the same code
runs on a cpu only machine or on a colab gpu without any change
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
