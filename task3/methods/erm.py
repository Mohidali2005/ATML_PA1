"""
This file does not train anything new. It loads the source only
checkpoint already trained in task two which is the erm baseline reused
unchanged here and saves an identical copy under the task three
checkpoint directory so every later script can treat it like any other
task three method
"""

import os

from task3.configs.config import load_config
from task3.methods.common import build_model,save_checkpoint,load_checkpoint

def main():
    """
    This function loads the task two source only checkpoint into a fresh
    backbone and head and writes it back out as the task three erm
    checkpoint
    """
    cfg = load_config("erm")
    backbone,head = build_model()
    task2_checkpoint = f"{cfg['paths']['task2_checkpoints_dir']}/source_only.pt"
    load_checkpoint(backbone,head,task2_checkpoint)

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/erm.pt"
    save_checkpoint(backbone,head,checkpoint_path)
    print(f"loaded task2 source only checkpoint and saved it as {checkpoint_path}")

if __name__ == "__main__":
    main()
