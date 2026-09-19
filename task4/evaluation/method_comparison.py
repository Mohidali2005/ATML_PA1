"""
this file builds the common comparison across vanilla, gcsc, and proser
using closed set accuracy together with near and far open set metrics.
every model is scored with mls, and proser gets a second row scored with
its own placeholder based detection score
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from task4.configs.config import load_config
from task4.data.cifar10 import NUM_KNOWN_CLASSES
from task4.extract_outputs import load_cache
from task4.scores.mls import mls_score
from task4.scores.proser import proser_score
from task4.evaluation.metrics import auroc,acceptance_rate
from task4.evaluation.thresholds import calibrate_threshold

def closed_set_accuracy(cache):
    """
    this function returns the top one accuracy over only the ten known
    class logits for one cached set of model outputs
    """
    preds = cache["logits"][:,:NUM_KNOWN_CLASSES].argmax(dim=1)
    return (preds==cache["labels"]).float().mean().item()

def evaluate_row(model_name,score_name,score_fn,val_cache,test_cache,near_cache,far_cache,target_tpr):
    """
    this function scores one model's validation, test, near, and far
    caches with the given score function and returns one comparison
    table row
    """
    val_scores = score_fn(val_cache)
    test_scores = score_fn(test_cache)
    near_scores = score_fn(near_cache)
    far_scores = score_fn(far_cache)
    threshold = calibrate_threshold(val_scores,target_tpr)
    all_unknown = np.concatenate([near_scores,far_scores])
    return {
        "model":model_name,
        "score":score_name,
        "csa":closed_set_accuracy(test_cache),
        "near_auroc":auroc(test_scores,near_scores),
        "far_auroc":auroc(test_scores,far_scores),
        "all_auroc":auroc(test_scores,all_unknown),
        "threshold":threshold,
        "test_acceptance_rate":acceptance_rate(test_scores,threshold),
        "near_acceptance_rate":acceptance_rate(near_scores,threshold),
        "far_acceptance_rate":acceptance_rate(far_scores,threshold),
    }

def plot_csa_vs_osr(table,out_path):
    """
    this function draws a grouped bar chart comparing closed set
    accuracy against near and far auroc for every method so the
    recognition versus rejection trade off is visible in one figure
    """
    labels = [f"{row.model}\n({row.score})" for row in table.itertuples()]
    x = np.arange(len(table))
    width = 0.25
    fig,ax = plt.subplots(figsize=(8,4.5))
    ax.bar(x-width,table["csa"],width,label="closed set accuracy",color="steelblue")
    ax.bar(x,table["near_auroc"],width,label="near auroc",color="darkorange")
    ax.bar(x+width,table["far_auroc"],width,label="far auroc",color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title("known class recognition versus unknown rejection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    """
    this function loads every cached model output, builds the vanilla,
    gcsc, and proser rows, and saves the common method comparison table
    and its bar chart
    """
    cfg = load_config("vanilla")
    target_tpr = cfg["evaluation"]["target_tpr"]

    mls_fn = lambda cache: mls_score(cache["logits"][:,:NUM_KNOWN_CLASSES]).numpy()

    rows = []
    for model_name in ["vanilla","gcsc","proser"]:
        val_cache = load_cache(cfg,model_name,"val")
        test_cache = load_cache(cfg,model_name,"test")
        near_cache = load_cache(cfg,model_name,"near")
        far_cache = load_cache(cfg,model_name,"far")
        row = evaluate_row(model_name,"mls",mls_fn,val_cache,test_cache,near_cache,far_cache,target_tpr)
        print(f"{model_name} mls csa {row['csa']:.3f} near_auroc {row['near_auroc']:.3f} far_auroc {row['far_auroc']:.3f}")
        rows.append(row)

    placeholder_fn = lambda cache: proser_score(cache["logits"],NUM_KNOWN_CLASSES).numpy()
    proser_val = load_cache(cfg,"proser","val")
    proser_test = load_cache(cfg,"proser","test")
    proser_near = load_cache(cfg,"proser","near")
    proser_far = load_cache(cfg,"proser","far")
    row = evaluate_row("proser","placeholder",placeholder_fn,proser_val,proser_test,proser_near,proser_far,target_tpr)
    print(f"proser placeholder csa {row['csa']:.3f} near_auroc {row['near_auroc']:.3f} far_auroc {row['far_auroc']:.3f}")
    rows.append(row)

    table = pd.DataFrame(rows)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)
    table.to_csv(f"{tables_dir}/method_comparison.csv",index=False)

    plot_csa_vs_osr(table,f"{figures_dir}/csa_vs_osr_bars.png")
    return table

if __name__ == "__main__":
    main()
