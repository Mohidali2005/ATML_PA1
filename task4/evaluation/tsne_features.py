"""
this file visualizes the penultimate feature space of the vanilla and
proser models together with the near and far unknown images, to see
whether proser's placeholders visibly separate known clusters from
unknown examples more than the vanilla model does
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from task4.configs.config import load_config,set_seed
from task4.data.cifar10 import CLASSES
from task4.extract_outputs import load_cache

def build_feature_pool(cfg,model_name,n_known_per_class,rng):
    """
    this function stacks a class balanced subset of known test features
    together with every near and far unknown feature for one model and
    returns the features, their group label, and their fine grained
    class name
    """
    test_cache = load_cache(cfg,model_name,"test")
    near_cache = load_cache(cfg,model_name,"near")
    far_cache = load_cache(cfg,model_name,"far")

    labels = test_cache["labels"].numpy()
    chosen = []
    for c in range(len(CLASSES)):
        class_idx = np.where(labels==c)[0]
        chosen.extend(rng.choice(class_idx,size=n_known_per_class,replace=False))
    chosen = np.array(chosen)

    known_feats = test_cache["features"][chosen].numpy()
    known_group = ["known"]*len(chosen)

    near_feats = near_cache["features"].numpy()
    far_feats = far_cache["features"].numpy()

    feats = np.concatenate([known_feats,near_feats,far_feats],axis=0)
    group = known_group+["near"]*len(near_cache["names"])+["far"]*len(far_cache["names"])
    return feats,group

def plot_tsne(points,group,out_path,title):
    """
    this function draws one scatter plot of the two dimensional t sne
    projection colored by whether each point is a known, near unknown,
    or far unknown example
    """
    colors = {"known":"steelblue","near":"darkorange","far":"firebrick"}
    fig,ax = plt.subplots(figsize=(6,5.5))
    group = np.array(group)
    for group_name in ["known","near","far"]:
        mask = group==group_name
        ax.scatter(points[mask,0],points[mask,1],s=10,alpha=0.6,label=group_name,color=colors[group_name])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    """
    this function fits one t sne projection per model on its own pooled
    known and unknown features and saves a comparison figure for vanilla
    against proser
    """
    cfg = load_config("vanilla")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir,exist_ok=True)

    for model_name in ["vanilla","proser"]:
        feats,group = build_feature_pool(cfg,model_name,cfg["evaluation"]["tsne_known_per_class"],rng)
        tsne = TSNE(n_components=2,perplexity=30,random_state=cfg["seed"])
        points = tsne.fit_transform(feats)
        out_path = f"{figures_dir}/feature_tsne_{model_name}.png"
        plot_tsne(points,group,out_path,f"{model_name} known versus unknown feature space")
        print(f"saved {out_path}")

if __name__ == "__main__":
    main()
