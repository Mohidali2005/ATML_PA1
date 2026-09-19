"""
this file compares the four post hoc novelty scores on the same frozen
vanilla model. it saves the required comparison table together with a
compact score distribution figure and two additional figures, an auroc
bar chart and roc curves, that make the differences between scores easy
to read
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from task4.configs.config import load_config
from task4.extract_outputs import load_cache
from task4.scores.msp import msp_score
from task4.scores.mls import mls_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import fit_mahalanobis,mahalanobis_score
from task4.evaluation.metrics import auroc,acceptance_rate
from task4.evaluation.thresholds import calibrate_threshold

SCORE_NAMES = ["msp","mls","energy","mahalanobis"]
SCORE_COLORS = {"msp":"steelblue","mls":"darkorange","energy":"seagreen","mahalanobis":"firebrick"}

def compute_scores(cache,mahalanobis_fit):
    """
    this function turns one cached set of logits and features into a
    dictionary of unknownness scores for all four post hoc methods
    """
    logits = cache["logits"]
    feats = cache["features"].numpy()
    means,inv_variance = mahalanobis_fit
    return {
        "msp":msp_score(logits).numpy(),
        "mls":mls_score(logits).numpy(),
        "energy":energy_score(logits).numpy(),
        "mahalanobis":mahalanobis_score(feats,means,inv_variance),
    }

def plot_score_distributions(known_scores,near_scores,far_scores,out_path):
    """
    this function draws the compact three panel score distribution
    figure for msp, mls, and mahalanobis, overlaying known, near, and
    far histograms on each panel
    """
    panel_scores = ["msp","mls","mahalanobis"]
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for ax,score_name in zip(axes,panel_scores):
        ax.hist(known_scores[score_name],bins=40,density=True,alpha=0.5,label="known",color="steelblue")
        ax.hist(near_scores[score_name],bins=40,density=True,alpha=0.5,label="near unknown",color="darkorange")
        ax.hist(far_scores[score_name],bins=40,density=True,alpha=0.5,label="far unknown",color="firebrick")
        ax.set_title(score_name)
        ax.set_xlabel("unknownness score")
        ax.set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_auroc_bars(table,out_path):
    """
    this function draws a grouped bar chart comparing near, far, and all
    unknown auroc across the four post hoc scores so the ranking between
    scores is easy to read at a glance
    """
    x = np.arange(len(table))
    width = 0.25
    fig,ax = plt.subplots(figsize=(7,4.5))
    ax.bar(x-width,table["near_auroc"],width,label="near",color="darkorange")
    ax.bar(x,table["far_auroc"],width,label="far",color="firebrick")
    ax.bar(x+width,table["all_auroc"],width,label="all",color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(table["score"])
    ax.set_ylabel("auroc")
    ax.set_ylim(0,1)
    ax.axhline(0.5,color="gray",linestyle="--",linewidth=1)
    ax.set_title("vanilla model: auroc by post hoc score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_roc_curves(known_scores,near_scores,far_scores,out_path):
    """
    this function draws roc curves for every post hoc score across near,
    far, and all unknowns in one compact three panel figure
    """
    groups = {"near":near_scores,"far":far_scores}
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    for ax,(group_name,unknown_scores) in zip(axes[:2],groups.items()):
        for score_name in SCORE_NAMES:
            labels = np.concatenate([np.zeros(len(known_scores[score_name])),np.ones(len(unknown_scores[score_name]))])
            scores = np.concatenate([known_scores[score_name],unknown_scores[score_name]])
            fpr,tpr,_ = roc_curve(labels,scores)
            ax.plot(fpr,tpr,label=score_name,color=SCORE_COLORS[score_name])
        ax.plot([0,1],[0,1],color="gray",linestyle="--",linewidth=1)
        ax.set_title(f"{group_name} unknown")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")

    ax = axes[2]
    for score_name in SCORE_NAMES:
        all_unknown = np.concatenate([near_scores[score_name],far_scores[score_name]])
        labels = np.concatenate([np.zeros(len(known_scores[score_name])),np.ones(len(all_unknown))])
        scores = np.concatenate([known_scores[score_name],all_unknown])
        fpr,tpr,_ = roc_curve(labels,scores)
        ax.plot(fpr,tpr,label=score_name,color=SCORE_COLORS[score_name])
    ax.plot([0,1],[0,1],color="gray",linestyle="--",linewidth=1)
    ax.set_title("all unknown")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    """
    this function fits mahalanobis on the vanilla training features,
    then scores the validation, test, near, and far caches with every
    post hoc score, saves the comparison table, and plots the score
    distributions, the auroc bar chart, and the roc curves
    """
    cfg = load_config("vanilla")
    train_cache = load_cache(cfg,"vanilla","train")
    mahalanobis_fit = fit_mahalanobis(train_cache["features"].numpy(),train_cache["labels"].numpy(),10)

    val_cache = load_cache(cfg,"vanilla","val")
    test_cache = load_cache(cfg,"vanilla","test")
    near_cache = load_cache(cfg,"vanilla","near")
    far_cache = load_cache(cfg,"vanilla","far")

    val_scores = compute_scores(val_cache,mahalanobis_fit)
    test_scores = compute_scores(test_cache,mahalanobis_fit)
    near_scores = compute_scores(near_cache,mahalanobis_fit)
    far_scores = compute_scores(far_cache,mahalanobis_fit)

    target_tpr = cfg["evaluation"]["target_tpr"]
    rows = []
    for score_name in SCORE_NAMES:
        threshold = calibrate_threshold(val_scores[score_name],target_tpr)
        all_unknown = np.concatenate([near_scores[score_name],far_scores[score_name]])
        row = {
            "score":score_name,
            "near_auroc":auroc(test_scores[score_name],near_scores[score_name]),
            "far_auroc":auroc(test_scores[score_name],far_scores[score_name]),
            "all_auroc":auroc(test_scores[score_name],all_unknown),
            "threshold":threshold,
            "test_acceptance_rate":acceptance_rate(test_scores[score_name],threshold),
            "near_acceptance_rate":acceptance_rate(near_scores[score_name],threshold),
            "far_acceptance_rate":acceptance_rate(far_scores[score_name],threshold),
        }
        print(f"{score_name} near_auroc {row['near_auroc']:.3f} far_auroc {row['far_auroc']:.3f} all_auroc {row['all_auroc']:.3f}")
        rows.append(row)

    table = pd.DataFrame(rows)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)
    table.to_csv(f"{tables_dir}/vanilla_score_comparison.csv",index=False)

    plot_score_distributions(test_scores,near_scores,far_scores,f"{figures_dir}/score_distributions.png")
    plot_auroc_bars(table,f"{figures_dir}/auroc_by_score_bars.png")
    plot_roc_curves(test_scores,near_scores,far_scores,f"{figures_dir}/roc_curves.png")

if __name__ == "__main__":
    main()
