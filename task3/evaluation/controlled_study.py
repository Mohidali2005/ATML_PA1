"""
This file runs the controlled design study for task three. It repeats
sam training at three different perturbation radii while keeping every
other setting fixed and reports how source performance the sharpness
proxy and sketch performance move with the perturbation strength
"""

import os
import pandas as pd

from task3.configs.config import load_config
from task3.methods.sam import run_training
from task3.methods.common import load_data,build_model,load_checkpoint
from task3.evaluation.domain_metrics import evaluate,mean_and_worst,load_target_eval_loader
from task3.evaluation.sharpness import build_sharpness_batch,sharpness_score
from shared.pacs_protocol import SOURCE_DOMAINS

def main():
    """
    This function trains sam once for every candidate perturbation
    radius then evaluates each resulting checkpoint and saves a compact
    comparison table
    """
    cfg = load_config("sam")
    rhos = cfg["sam"]["controlled_rhos"]

    df,splits,_,val_loaders = load_data(cfg)
    target_loader = load_target_eval_loader(df,splits,cfg)
    sharp_images,sharp_labels = build_sharpness_batch(df,splits,cfg)

    rows = []
    for rho in rhos:
        checkpoint_name = f"sam_rho_{rho}"
        run_training(cfg,rho,checkpoint_name)

        backbone,head = build_model()
        load_checkpoint(backbone,head,f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt")

        source_accs = []
        for domain in SOURCE_DOMAINS:
            accuracy,_ = evaluate(backbone,head,val_loaders[domain])
            source_accs.append(accuracy)
        mean_source_acc,_ = mean_and_worst(source_accs)
        target_acc,_ = evaluate(backbone,head,target_loader)
        sharpness = sharpness_score(backbone,head,sharp_images,sharp_labels,cfg)

        row = {"rho":rho,"mean_source_acc":mean_source_acc,"sharpness":sharpness,"sketch_acc":target_acc}
        print(row)
        rows.append(row)

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['tables_dir']}/sam_rho_study.csv",index=False)

if __name__ == "__main__":
    main()
