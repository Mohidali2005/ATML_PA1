"""
this file saves and plots the per epoch training curves shared by every
training run in task four so each checkpoint leaves behind evidence of
how it trained
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def save_training_curve(history,name,cfg,alignment_key=None,alignment_label=None):
    """
    this function saves the given list of per epoch metric dictionaries
    to a csv table and plots the training loss and the validation
    accuracy for the given run alongside an optional extra curve such as
    proser's data placeholder loss
    """
    table = pd.DataFrame(history)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)
    table.to_csv(f"{tables_dir}/{name}_training_curve.csv",index=False)

    num_plots = 3 if alignment_key else 2
    fig,axes = plt.subplots(1,num_plots,figsize=(5*num_plots,4))

    axes[0].plot(table["epoch"],table["loss"],color="steelblue")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].set_title(f"{name} training loss")

    axes[1].plot(table["epoch"],table["val_acc"],color="darkorange")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title(f"{name} validation accuracy")

    if alignment_key:
        axes[2].plot(table["epoch"],table[alignment_key],color="firebrick")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel(alignment_label)
        axes[2].set_title(f"{name} {alignment_label}")

    fig.tight_layout()
    fig.savefig(f"{figures_dir}/{name}_training_curve.png")
    plt.close(fig)
