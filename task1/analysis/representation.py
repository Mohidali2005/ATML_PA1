"""
This file fits one t sne projection per backbone using the clean eval
features together with a set of transformed features so both clean
and transformed examples can be seen in the same space
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from configs.config import load_config,set_seed

def build_combined_pool(backbone_name,features_dir,subset_idx,eval_labels,classes,cue_meta):
    """
    This function stacks the clean features together with the
    grayscale features the patch shuffled features one translation
    variant and the cue conflict features. It returns the stacked
    features their class labels and a clean or transformed marker for
    every point
    """
    class_to_id = {name:i for i,name in enumerate(classes)}

    clean_feats = torch.load(f"{features_dir}/{backbone_name}_eval_clean.pt")[subset_idx]
    grayscale_feats = torch.load(f"{features_dir}/{backbone_name}_eval_grayscale.pt")[subset_idx]
    patch_feats = torch.load(f"{features_dir}/{backbone_name}_eval_patchshuffle.pt")[subset_idx]
    translate_feats = torch.load(f"{features_dir}/{backbone_name}_eval_translate_d32_right.pt")[subset_idx]
    cue_feats = torch.load(f"{features_dir}/{backbone_name}_cue_conflict.pt")

    subset_labels = eval_labels[subset_idx]
    cue_labels = np.array([class_to_id[item["content_class"]] for item in cue_meta["items"]])

    feats = torch.cat([clean_feats,grayscale_feats,patch_feats,translate_feats,cue_feats],dim=0).numpy()
    labels = np.concatenate([subset_labels,subset_labels,subset_labels,subset_labels,cue_labels])

    n_subset = len(subset_idx)
    n_cue = len(cue_feats)
    marker = ["clean"]*n_subset+["transformed"]*(n_subset*3)+["transformed"]*n_cue

    return feats,labels,marker

def plot_projection(points,labels,marker,classes,title,out_path):
    """
    This function draws one scatter plot of the two dimensional
    projection. Color shows the ground truth class and marker style
    shows whether a point is clean or transformed
    """
    class_names = [classes[label] for label in labels]
    fig,ax = plt.subplots(figsize=(7,6))
    sns.scatterplot(
        x=points[:,0],y=points[:,1],
        hue=class_names,style=marker,
        palette="tab10",markers={"clean":"o","transformed":"X"},
        s=25,alpha=0.7,ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def main():
    """
    This function builds the combined feature pool for every backbone
    fits a t sne projection on it and saves the resulting plot
    """
    cfg = load_config()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    features_dir = cfg["paths"]["cache_dir"]+"/features"
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir,exist_ok=True)

    with open(cfg["paths"]["cache_dir"]+"/splits.json") as f:
        splits = json.load(f)
    classes = splits["classes"]

    with open(cfg["paths"]["cache_dir"]+"/cue_conflict_meta.json") as f:
        cue_meta = json.load(f)

    eval_labels = np.load(features_dir+"/labels.npz")["eval_labels"]

    rep_cfg = cfg["representation"]
    subset_size = min(rep_cfg["subset_size"],len(eval_labels))
    subset_idx = rng.choice(len(eval_labels),size=subset_size,replace=False)

    for backbone_name in ["resnet","vit","clip"]:
        feats,labels,marker = build_combined_pool(backbone_name,features_dir,subset_idx,eval_labels,classes,cue_meta)

        tsne = TSNE(
            n_components=2,
            perplexity=rep_cfg["perplexity"],
            max_iter=rep_cfg["max_iter"],
            learning_rate=rep_cfg["learning_rate"],
            random_state=cfg["seed"],
        )
        points = tsne.fit_transform(feats)

        out_path = f"{figures_dir}/tsne_{backbone_name}.png"
        plot_projection(points,labels,marker,classes,f"{backbone_name} clean versus transformed",out_path)
        print(f"saved {out_path}")

if __name__ == "__main__":
    main()
