"""
This file runs the shared training loop used by every task three method
that actually trains. It evaluates the mean source validation macro f1
after every epoch then saves the best checkpoint and stops early once
performance stops improving all without ever touching sketch
"""

from task3.evaluation.domain_metrics import evaluate
from task3.methods.common import save_checkpoint
from shared.pacs_protocol import SOURCE_DOMAINS

def train_with_source_selection(backbone,head,val_loaders,cfg,checkpoint_path,train_epoch):
    """
    This function repeatedly calls the given train epoch function then
    checks the resulting mean source validation macro f1 after every
    epoch and saves the backbone and head whenever that score improves
    and returns the full per epoch history once training stops
    """
    best_mean_f1 = -1
    epochs_without_improve = 0
    history = []

    for epoch in range(cfg["optimizer"]["max_epochs"]):
        epoch_metrics = train_epoch()

        val_f1s = []
        for domain in SOURCE_DOMAINS:
            accuracy,macro_f1 = evaluate(backbone,head,val_loaders[domain])
            val_f1s.append(macro_f1)
            print(f"epoch {epoch} {domain} val_acc {accuracy:.3f} val_f1 {macro_f1:.3f}")
        mean_f1 = sum(val_f1s)/len(val_f1s)

        epoch_metrics["epoch"] = epoch
        epoch_metrics["mean_val_f1"] = mean_f1
        history.append(epoch_metrics)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            save_checkpoint(backbone,head,checkpoint_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg["optimizer"]["patience"]:
            print(f"stopping early at epoch {epoch} best mean_f1 {best_mean_f1:.3f}")
            break

    return history
