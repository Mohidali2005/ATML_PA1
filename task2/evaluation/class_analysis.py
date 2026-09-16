"""
This file uses target labels for the first time to look at per class
accuracy on sketch for every trained method compared with source only
and to inspect which classes get confused with each other
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix

from task2.configs.config import load_config,set_seed
from task2.methods.common import load_data,load_target_eval_loader,build_model,load_checkpoint
from task2.evaluation.metrics import predict,per_class_accuracy
from shared.pacs import CLASSES

METHODS = ["source_only","dan","dann","cdan"]

def main():
    """
    This function computes per class target accuracy for every trained
    method saves a comparison table against source only and saves one
    confusion matrix per method
    """
    cfg = load_config("source_only")
    set_seed(cfg["seed"])
    df,splits,_,_ = load_data(cfg)
    target_loader = load_target_eval_loader(df,splits,cfg)

    per_class_rows = {}
    os.makedirs(cfg["paths"]["tables_dir"],exist_ok=True)
    for method in METHODS:
        backbone,head = build_model()
        checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{method}.pt"
        load_checkpoint(backbone,head,checkpoint_path)

        accuracies = per_class_accuracy(backbone,head,target_loader,len(CLASSES))
        per_class_rows[method] = accuracies
        print(f"{method} per class accuracy {accuracies}")

        preds,labels = predict(backbone,head,target_loader)
        matrix = confusion_matrix(labels.numpy(),preds.numpy(),labels=list(range(len(CLASSES))))
        confusion_table = pd.DataFrame(matrix,index=CLASSES,columns=CLASSES)
        confusion_table.to_csv(f"{cfg['paths']['tables_dir']}/{method}_confusion_matrix.csv")

    table = pd.DataFrame(per_class_rows)
    table.index = CLASSES
    for method in METHODS:
        if method != "source_only":
            table[f"{method}_delta"] = table[method]-table["source_only"]
    table.to_csv(f"{cfg['paths']['tables_dir']}/per_class_target_accuracy.csv")

if __name__ == "__main__":
    main()
