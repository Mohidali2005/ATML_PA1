"""
This file builds the final task three comparison table across erm dan
dg and sam by combining source validation performance with the source
domain separability and sharpness diagnostics then loads sketch for the
first time to compute target performance and the per class analysis
"""

import os
import pandas as pd

from task3.configs.config import load_config,set_seed
from task3.methods.common import load_data,build_model,load_checkpoint
from task3.evaluation.domain_metrics import evaluate,mean_and_worst,load_target_eval_loader
from task3.evaluation.source_domain_separability import collect_balanced_source_features,source_domain_separability_score
from task3.evaluation.sharpness import build_sharpness_batch,sharpness_score
from task3.evaluation import class_analysis
from shared.pacs_protocol import SOURCE_DOMAINS

METHODS = ["erm","dan_dg","sam"]

def main():
    """
    This function evaluates every trained method on every source
    validation domain and on sketch then computes the source domain
    separability and sharpness diagnostics for each one and saves the
    resulting master comparison table
    """
    cfg = load_config("erm")
    set_seed(cfg["seed"])
    df,splits,_,val_loaders = load_data(cfg)
    target_loader = load_target_eval_loader(df,splits,cfg)
    sharp_images,sharp_labels = build_sharpness_batch(df,splits,cfg)

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
        row["mean_source_acc"],row["worst_source_acc"] = mean_and_worst(source_accs)
        row["mean_source_f1"],row["worst_source_f1"] = mean_and_worst(source_f1s)

        target_acc,target_f1 = evaluate(backbone,head,target_loader)
        row["sketch_acc"] = target_acc
        row["sketch_f1"] = target_f1
        target_accuracies[method] = target_acc

        balanced_feats = collect_balanced_source_features(backbone,val_loaders,cfg)
        row["source_domain_separability"] = source_domain_separability_score(balanced_feats,cfg)
        row["sharpness"] = sharpness_score(backbone,head,sharp_images,sharp_labels,cfg)

        rows.append(row)
        print(f"{method} mean_source_acc {row['mean_source_acc']:.3f} sketch_acc {target_acc:.3f} source_domain_separability {row['source_domain_separability']:.3f} sharpness {row['sharpness']:.4f}")

    table = pd.DataFrame(rows)
    table["sketch_acc_change"] = table["sketch_acc"]-target_accuracies["erm"]

    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    table.to_csv(f"{cfg['paths']['tables_dir']}/method_comparison.csv",index=False)

    class_analysis.main()

if __name__ == "__main__":
    main()
