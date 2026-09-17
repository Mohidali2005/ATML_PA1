"""
This file uses sketch labels for the first time to look at per class
accuracy for every trained task three method compared with erm to
inspect which classes get confused with each other and to line the
results up against the matching task two numbers for the same classes
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix

from task3.configs.config import load_config,set_seed
from task3.methods.common import load_data,build_model,load_checkpoint
from task3.evaluation.domain_metrics import predict,per_class_accuracy,load_target_eval_loader
from shared.pacs import CLASSES

METHODS = ["erm","dan_dg","sam"]

def main():
    """
    This function computes per class sketch accuracy for every trained
    task three method then saves a comparison table against erm and
    against the matching task two numbers and saves one confusion matrix
    per method
    """
    cfg = load_config("erm")
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
        if method != "erm":
            table[f"{method}_delta"] = table[method]-table["erm"]

    task2_table = pd.read_csv(f"{cfg['paths']['task2_tables_dir']}/per_class_target_accuracy.csv",index_col=0)
    for column in ["source_only","dan","dann","cdan"]:
        table[f"task2_{column}"] = task2_table[column]

    table.to_csv(f"{cfg['paths']['tables_dir']}/per_class_target_accuracy.csv")

if __name__ == "__main__":
    main()
