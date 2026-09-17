"""
This file saves and plots the per epoch training curves shared by every
method in task three so each run leaves behind evidence of whether it
trained as intended
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def save_training_curve(history,name,cfg,alignment_key=None,alignment_label=None):
    """
    This function saves the given list of per epoch metric dictionaries
    to a csv table and plots the classification loss and the mean source
    validation macro f1 for the given run alongside an optional
    alignment metric such as the dan dg mmd penalty
    """
    table = pd.DataFrame(history)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)
    table.to_csv(f"{tables_dir}/{name}_training_curve.csv",index=False)

    num_plots = 3 if alignment_key else 2
    fig,axes = plt.subplots(1,num_plots,figsize=(5*num_plots,4))

    axes[0].plot(table["epoch"],table["cls_loss"],color="steelblue")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("classification loss")
    axes[0].set_title(f"{name} classification loss")

    axes[1].plot(table["epoch"],table["mean_val_f1"],color="darkorange")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mean source validation macro f1")
    axes[1].set_title(f"{name} validation macro f1")

    if alignment_key:
        axes[2].plot(table["epoch"],table[alignment_key],color="firebrick")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel(alignment_label)
        axes[2].set_title(f"{name} {alignment_label}")

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{name}_training_curve.png")
    plt.close(fig)
