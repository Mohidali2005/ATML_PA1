"""
This file evaluates the stabilized dann and cdan checkpoints on every
source validation domain and on sketch and prints them side by side with
the original unstabilized numbers already saved in
task2/results/tables/method_comparison.csv so the two can be compared
directly without touching that original table
"""

import os
import numpy as np
import pandas as pd

from task2.configs.config import load_config,set_seed,TASK2_ROOT
from task2.methods.common import load_data,load_target_eval_loader,build_model,load_checkpoint
from task2.evaluation.metrics import evaluate
from task2.evaluation.domain_separability import collect_domain_features,domain_separability_score
from task2.experiments.dann_stabilized import stabilized_paths
from shared.pacs_protocol import SOURCE_DOMAINS

METHODS = ["dann_stabilized","cdan_stabilized"]

def main():
    """
    This function evaluates every stabilized checkpoint the same way
    evaluate_final does for the original methods then prints a comparison
    against the matching original row so the stability test's effect is
    visible directly
    """
    cfg = load_config("source_only")
    cfg = stabilized_paths(cfg)
    set_seed(cfg["seed"])
    df,splits,_,val_loaders = load_data(cfg)
    target_loader = load_target_eval_loader(df,splits,cfg)
    rng = np.random.default_rng(cfg["domain_separability"]["seed"])

    original = pd.read_csv(f"{TASK2_ROOT}/results/tables/method_comparison.csv").set_index("method")

    rows = []
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

        source_feats,target_feats = collect_domain_features(backbone,df,splits,cfg,rng)
        row["domain_separability"] = domain_separability_score(source_feats,target_feats,cfg)
        rows.append(row)

        original_method = method.replace("_stabilized","")
        original_row = original.loc[original_method]
        print(f"{method} mean_source_f1 {row['mean_source_f1']:.3f} target_acc {row['target_acc']:.3f} domain_separability {row['domain_separability']:.3f}")
        print(f"{original_method} original mean_source_f1 {original_row['mean_source_f1']:.3f} target_acc {original_row['target_acc']:.3f} domain_separability {original_row['domain_separability']:.3f}")

    tables_dir = cfg["paths"]["tables_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{tables_dir}/stabilized_comparison.csv",index=False)

if __name__ == "__main__":
    main()
