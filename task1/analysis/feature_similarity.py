"""
This file measures how much each backbone representation moves under
every required intervention. It compares the clean feature and the
transformed feature of the same image using cosine similarity
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from configs.config import load_config

def cosine_stability(clean_feats,transformed_feats):
    """
    This function returns the average cosine similarity between each
    clean feature vector and its matching transformed feature vector
    A value close to one means the representation barely moved
    """
    clean_norm = clean_feats/clean_feats.norm(dim=1,keepdim=True)
    transformed_norm = transformed_feats/transformed_feats.norm(dim=1,keepdim=True)
    similarity = (clean_norm*transformed_norm).sum(dim=1)
    return float(similarity.mean())

def cue_conflict_stability(backbone_name,features_dir,cue_meta):
    """
    This function measures how far the cue conflict representation has
    moved away from its own original content image representation
    The cue conflict images are not in the same order as the eval
    images so the matching content index from the metadata is used
    instead
    """
    clean_feats = torch.load(f"{features_dir}/{backbone_name}_eval_clean.pt")
    conflict_feats = torch.load(f"{features_dir}/{backbone_name}_cue_conflict.pt")
    content_indices = [item["content_eval_index"] for item in cue_meta["items"]]
    matched_clean_feats = clean_feats[content_indices]
    return cosine_stability(matched_clean_feats,conflict_feats)

def plot_feature_stability(table,figures_dir):
    """
    This function draws one bar chart showing the cosine stability of
    every backbone across every required intervention so the amount
    of representation movement is easy to compare at a glance
    """
    backbone_colors = {"resnet":"steelblue","vit":"darkorange","clip":"seagreen"}
    condition_order = [
        "eval_grayscale",
        "eval_colorswap",
        "eval_patchshuffle",
        "translate_d8",
        "translate_d16",
        "translate_d32",
        "cue_conflict",
    ]
    condition_labels = {
        "eval_grayscale":"grayscale",
        "eval_colorswap":"color swap",
        "eval_patchshuffle":"patch shuffle",
        "translate_d8":"translate 8",
        "translate_d16":"translate 16",
        "translate_d32":"translate 32",
        "cue_conflict":"cue conflict",
    }

    fig,ax = plt.subplots(figsize=(10,5))
    bar_width = 0.25
    x = np.arange(len(condition_order))
    for i,backbone_name in enumerate(backbone_colors):
        values = [
            table[(table["backbone"]==backbone_name)&(table["condition"]==c)]["cosine_stability"].iloc[0]
            for c in condition_order
        ]
        ax.bar(x+i*bar_width,values,width=bar_width,label=backbone_name,color=backbone_colors[backbone_name])

    ax.set_xticks(x+bar_width)
    ax.set_xticklabels([condition_labels[c] for c in condition_order],rotation=20)
    ax.set_ylabel("cosine stability")
    ax.set_ylim(0,1.05)
    ax.legend()
    plt.tight_layout()

    out_path = figures_dir+"/feature_stability.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")

def main():
    """
    This function loads the cached features for every backbone and
    every required intervention builds one stability table and saves
    it to the results folder
    """
    cfg = load_config()
    features_dir = cfg["paths"]["cache_dir"]+"/features"
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)

    with open(cfg["paths"]["cache_dir"]+"/cue_conflict_meta.json") as f:
        cue_meta = json.load(f)

    translation_cfg = cfg["interventions"]["translation"]
    displacements = translation_cfg["displacements"]
    directions = translation_cfg["directions"]

    rows = []
    for backbone_name in ["resnet","vit","clip"]:
        clean_feats = torch.load(f"{features_dir}/{backbone_name}_eval_clean.pt")

        for condition in ["eval_grayscale","eval_colorswap","eval_patchshuffle"]:
            transformed_feats = torch.load(f"{features_dir}/{backbone_name}_{condition}.pt")
            score = cosine_stability(clean_feats,transformed_feats)
            rows.append({"backbone":backbone_name,"condition":condition,"cosine_stability":score})

        for d in displacements:
            if d == 0:
                keys = ["eval_translate_d0"]
            else:
                keys = [f"eval_translate_d{d}_{direction}" for direction in directions]
            scores = []
            for key in keys:
                transformed_feats = torch.load(f"{features_dir}/{backbone_name}_{key}.pt")
                scores.append(cosine_stability(clean_feats,transformed_feats))
            rows.append({"backbone":backbone_name,"condition":f"translate_d{d}","cosine_stability":float(np.mean(scores))})

        score = cue_conflict_stability(backbone_name,features_dir,cue_meta)
        rows.append({"backbone":backbone_name,"condition":"cue_conflict","cosine_stability":score})

    table = pd.DataFrame(rows)
    table.to_csv(f"{tables_dir}/feature_stability.csv",index=False)
    print(table)

    plot_feature_stability(table,figures_dir)

if __name__ == "__main__":
    main()
