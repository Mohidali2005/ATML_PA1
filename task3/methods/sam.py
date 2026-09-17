"""
This file trains the resnet eighteen backbone and its linear head with
standard sharpness aware minimization on top of the ordinary source
cross entropy loss. It never removes domain information on purpose and
only tests whether a locally stable solution transfers better on its own
"""

import os
import torch
import torch.nn as nn

from task3.configs.config import load_config,set_seed
from task3.methods.common import load_data,build_model
from task3.selection.source_validation import train_with_source_selection
from task3.evaluation.history import save_training_curve
from shared.bn_utils import freeze_batchnorm
from shared.pacs_protocol import SOURCE_DOMAINS,cycle
from shared.device import DEVICE

class SAM:
    """
    This class wraps a base optimizer with the sharpness aware
    minimization update rule. The first step finds a normalized ascent
    perturbation from the current gradient and the second step removes
    that perturbation then applies the base optimizer update using the
    gradient computed at the perturbed point
    """

    def __init__(self,params,base_optimizer,rho):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.perturbations = []

    def first_step(self):
        """
        This method perturbs every parameter along its own current
        gradient direction scaled so the perturbation as a whole has
        norm rho
        """
        grads = [p.grad for p in self.params if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]),2)
        scale = self.rho/(grad_norm+1e-12)
        self.perturbations = []
        for p in self.params:
            if p.grad is None:
                self.perturbations.append(None)
                continue
            perturbation = p.grad*scale
            p.data.add_(perturbation)
            self.perturbations.append(perturbation)

    def second_step(self):
        """
        This method undoes the perturbation applied in the first step and
        takes the base optimizer step using the gradient computed at the
        perturbed point
        """
        for p,perturbation in zip(self.params,self.perturbations):
            if perturbation is not None:
                p.data.sub_(perturbation)
        self.base_optimizer.step()

    def zero_grad(self):
        """
        This method clears the gradients on the wrapped base optimizer
        """
        self.base_optimizer.zero_grad()

def make_train_epoch(backbone,head,train_loaders,sam,criterion,steps_per_epoch):
    """
    This function returns a train epoch closure that runs the two pass
    sharpness aware update on one domain balanced batch per step and
    reports the average classification loss measured at the original
    unperturbed parameters
    """
    def train_epoch():
        backbone.train()
        head.train()
        freeze_batchnorm(backbone)
        iterators = {domain:cycle(train_loaders[domain]) for domain in SOURCE_DOMAINS}
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            images = []
            labels = []
            for domain in SOURCE_DOMAINS:
                batch_images,batch_labels,_ = next(iterators[domain])
                images.append(batch_images)
                labels.append(batch_labels)
            images = torch.cat(images,dim=0).to(DEVICE)
            labels = torch.cat(labels,dim=0).to(DEVICE)

            sam.zero_grad()
            logits = head(backbone(images))
            loss = criterion(logits,labels)
            loss.backward()
            sam.first_step()

            sam.zero_grad()
            logits = head(backbone(images))
            criterion(logits,labels).backward()
            sam.second_step()

            total_loss += loss.item()
        return {"cls_loss":total_loss/steps_per_epoch}
    return train_epoch

def run_training(cfg,rho,checkpoint_name):
    """
    This function runs a complete sam training run for the given
    perturbation radius and saves the best checkpoint under the given
    name
    """
    set_seed(cfg["seed"])
    df,splits,train_loaders,val_loaders = load_data(cfg)
    backbone,head = build_model()
    base_optimizer = torch.optim.AdamW(list(backbone.parameters())+list(head.parameters()),lr=cfg["optimizer"]["lr"],weight_decay=cfg["optimizer"]["weight_decay"])
    sam = SAM(list(backbone.parameters())+list(head.parameters()),base_optimizer,rho)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(train_loaders[domain]) for domain in SOURCE_DOMAINS)
    train_epoch = make_train_epoch(backbone,head,train_loaders,sam,criterion,steps_per_epoch)

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt"
    history = train_with_source_selection(backbone,head,val_loaders,cfg,checkpoint_path,train_epoch)

    save_training_curve(history,checkpoint_name,cfg)
    return history

def main():
    """
    This function runs the main sam comparison using the fixed
    perturbation radius from the config file
    """
    cfg = load_config("sam")
    run_training(cfg,cfg["sam"]["rho"],"sam")

if __name__ == "__main__":
    main()
