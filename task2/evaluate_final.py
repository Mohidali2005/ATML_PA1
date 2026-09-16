"""
This file builds the final comparison table across every trained method
by combining source validation performance target performance and the
domain separability diagnostic then also runs the per class analysis
"""

import os
import numpy as np
import pandas as pd

from task2.configs.config import load_config,set_seed
from task2.methods.common import load_data,load_target_eval_loader,build_model,load_checkpoint
from task2.evaluation.metrics import evaluate
from task2.evaluation.domain_separability import collect_domain_features,domain_separability_score
from task2.evaluation import class_analysis
from shared.pacs_protocol import SOURCE_DOMAINS

METHODS = ["source_only","dan","dann","cdan"]

def main():
    """
    This function evaluates every trained method on every source
    validation domain and on sketch computes the domain separability
    score for each one and saves the resulting master comparison table
    """
    cfg = load_config("source_only")
    set_seed(cfg["seed"])
    df,splits,_,val_loaders = load_data(cfg)
    target_loader = load_target_eval_loader(df,splits,cfg)
    rng = np.random.default_rng(cfg["domain_separability"]["seed"])

    rows = []
    target_accuracies = {}
    for method in METHODS:
        backbone,head = build_model()
        checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{method}.pt"
        load_checkpoint(backbone,head,checkpoint_path)

        row = {"method":method}
        source_accs = []
        source_f1s = []
        for domain in SOURCE_DOMAINS:
            accuracy,macro_f1 = evaluate(backbone,head,val_loaders[domain])
            row[f"{domain}_val_acc"] = accuracy
            row[f"{domain}_val_f1"] = macro_f1
            source_accs.append(accuracy)
            source_f1s.append(macro_f1)
        row["mean_source_acc"] = sum(source_accs)/len(source_accs)
        row["mean_source_f1"] = sum(source_f1s)/len(source_f1s)

        target_acc,target_f1 = evaluate(backbone,head,target_loader)
        row["target_acc"] = target_acc
        row["target_f1"] = target_f1
        target_accuracies[method] = target_acc

        source_feats,target_feats = collect_domain_features(backbone,df,splits,cfg,rng)
        row["domain_separability"] = domain_separability_score(source_feats,target_feats,cfg)

        rows.append(row)
        print(f"{method} mean_source_acc {row['mean_source_acc']:.3f} target_acc {target_acc:.3f} domain_separability {row['domain_separability']:.3f}")

    table = pd.DataFrame(rows)
    table["target_acc_change"] = table["target_acc"]-target_accuracies["source_only"]

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    table.to_csv(f"{cfg['paths']['tables_dir']}/method_comparison.csv",index=False)

    class_analysis.main()

if __name__ == "__main__":
    main()
