"""
This file runs the controlled design study for task two. It repeats dan
training at three different mmd loss weights while keeping every other
setting fixed and reports how source performance domain separability
and target performance move with the alignment pressure
"""

import os
import numpy as np
import pandas as pd

from task2.configs.config import load_config
from task2.methods.dan import run_training
from task2.methods.common import load_data,load_target_eval_loader,build_model,load_checkpoint
from task2.evaluation.metrics import evaluate
from task2.evaluation.domain_separability import collect_domain_features,domain_separability_score
from shared.pacs_protocol import SOURCE_DOMAINS

def main():
    """
    This function trains dan once for every candidate mmd loss weight
    evaluates each resulting checkpoint and saves a compact comparison
    table
    """
    cfg = load_config("dan")
    lambdas = cfg["dan"]["controlled_lambdas"]

    df,splits,_,val_loaders = load_data(cfg)
    target_loader = load_target_eval_loader(df,splits,cfg)
    rng = np.random.default_rng(cfg["domain_separability"]["seed"])

    rows = []
    for lambda_mmd in lambdas:
        checkpoint_name = f"dan_lambda_{lambda_mmd}"
        run_training(cfg,lambda_mmd,checkpoint_name)

        backbone,head = build_model()
        load_checkpoint(backbone,head,f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt")

        source_accs = []
        for domain in SOURCE_DOMAINS:
            accuracy,_ = evaluate(backbone,head,val_loaders[domain])
            source_accs.append(accuracy)
        target_acc,_ = evaluate(backbone,head,target_loader)
        source_feats,target_feats = collect_domain_features(backbone,df,splits,cfg,rng)
        separability = domain_separability_score(source_feats,target_feats,cfg)

        row = {"lambda_mmd":lambda_mmd,"mean_source_acc":sum(source_accs)/len(source_accs),"target_acc":target_acc,"domain_separability":separability}
        print(row)
        rows.append(row)

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{cfg['paths']['tables_dir']}/dan_lambda_study.csv",index=False)

if __name__ == "__main__":
    main()
