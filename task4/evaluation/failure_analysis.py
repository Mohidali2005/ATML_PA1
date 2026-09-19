"""
this file inspects individual near and far unknown images that the
vanilla model's mls threshold incorrectly accepts as a known class, and
also summarizes which known classes absorb unknown images across both
groups
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from task4.configs.config import load_config
from task4.data.cifar10 import CLASSES
from task4.extract_outputs import load_cache
from task4.scores.mls import mls_score
from task4.evaluation.thresholds import calibrate_threshold

def find_accepted_unknowns(cache,threshold,group,n_examples):
    """
    this function returns the n most confidently accepted unknown
    examples from one cached unknown split along with the known class
    each one was predicted as
    """
    scores = mls_score(cache["logits"]).numpy()
    preds = cache["logits"].argmax(dim=1).numpy()
    accepted = np.where(scores<=threshold)[0]
    ordered = accepted[np.argsort(scores[accepted])]
    chosen = ordered[:n_examples]
    rows = []
    for i in chosen:
        rows.append({
            "unknown_group":group,
            "true_cifar100_class":cache["names"][i],
            "predicted_known_class":CLASSES[preds[i]],
            "score":float(scores[i]),
            "threshold":float(threshold),
        })
    return rows

def absorption_counts(cache,threshold,group):
    """
    this function counts how many accepted images of every unknown class
    in the given group were predicted as each known class
    """
    scores = mls_score(cache["logits"]).numpy()
    preds = cache["logits"].argmax(dim=1).numpy()
    accepted = np.where(scores<=threshold)[0]
    rows = []
    for i in accepted:
        rows.append({"group":group,"unknown_class":cache["names"][i],"predicted_known_class":CLASSES[preds[i]]})
    return rows

def plot_absorption_heatmap(counts_df,out_path):
    """
    this function draws a heatmap of how many accepted unknown images of
    every unknown class landed on every known class prediction
    """
    pivot = pd.crosstab(counts_df["unknown_class"],counts_df["predicted_known_class"])
    pivot = pivot.reindex(columns=CLASSES,fill_value=0)
    fig,ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(pivot.values,cmap="Reds",aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns,rotation=45,ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i,j]
            if value>0:
                ax.text(j,i,int(value),ha="center",va="center",fontsize=7,color="black")
    ax.set_title("accepted unknown images by predicted known class")
    fig.colorbar(im,ax=ax,label="accepted image count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    """
    this function calibrates the vanilla mls threshold on validation
    data, then saves the informative near and far failure cases and the
    absorption heatmap
    """
    cfg = load_config("vanilla")
    val_cache = load_cache(cfg,"vanilla","val")
    threshold = calibrate_threshold(mls_score(val_cache["logits"]).numpy(),cfg["evaluation"]["target_tpr"])

    n_examples = cfg["evaluation"]["n_failure_examples"]
    rows = []
    counts_rows = []
    for group in ["near","far"]:
        cache = load_cache(cfg,"vanilla",group)
        rows.extend(find_accepted_unknowns(cache,threshold,group,n_examples))
        counts_rows.extend(absorption_counts(cache,threshold,group))

    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)

    pd.DataFrame(rows).to_csv(f"{tables_dir}/failure_cases.csv",index=False)
    counts_df = pd.DataFrame(counts_rows,columns=["group","unknown_class","predicted_known_class"])
    counts_df.to_csv(f"{tables_dir}/unknown_absorption_counts.csv",index=False)
    plot_absorption_heatmap(counts_df,f"{figures_dir}/unknown_absorption_heatmap.png")
    print(f"vanilla mls threshold {threshold:.4f}")

if __name__ == "__main__":
    main()
