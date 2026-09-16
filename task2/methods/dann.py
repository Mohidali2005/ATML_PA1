"""
This file trains the resnet eighteen backbone and its linear head using
dann style adversarial alignment. A small domain discriminator sits
behind a gradient reversal layer and tries to tell source features from
target features while the backbone is pushed to confuse it
"""

import os
import torch
import torch.nn as nn

from task2.configs.config import load_config,set_seed
from task2.methods.common import load_data,load_target_train_loader,build_model,save_checkpoint
from task2.evaluation.metrics import evaluate
from task2.evaluation.history import save_training_curve
from task2.models.domain_discriminator import DomainDiscriminator,grl_alpha
from shared.bn_utils import freeze_batchnorm
from shared.pacs_protocol import SOURCE_DOMAINS,cycle
from shared.device import DEVICE

def train_epoch(backbone,head,discriminator,train_loaders,target_iter,optimizer,criterion,max_alpha,global_step,total_steps,steps_per_epoch):
    """
    This function runs one epoch of dann training and returns the
    updated global step count together with the average classification
    loss the average domain loss and the average domain discriminator
    accuracy for the epoch
    """
    backbone.train()
    head.train()
    discriminator.train()
    freeze_batchnorm(backbone)
    iterators = {domain:cycle(train_loaders[domain]) for domain in SOURCE_DOMAINS}
    total_cls_loss = 0.0
    total_domain_loss = 0.0
    total_domain_acc = 0.0
    for _ in range(steps_per_epoch):
        images = []
        labels = []
        for domain in SOURCE_DOMAINS:
            batch_images,batch_labels,_ = next(iterators[domain])
            images.append(batch_images)
            labels.append(batch_labels)
        source_images = torch.cat(images,dim=0).to(DEVICE)
        source_labels = torch.cat(labels,dim=0).to(DEVICE)
        target_images,_,_ = next(target_iter)
        target_images = target_images.to(DEVICE)

        progress = global_step/total_steps
        alpha = grl_alpha(progress,max_alpha)

        optimizer.zero_grad()
        source_features = backbone(source_images)
        target_features = backbone(target_images)
        logits = head(source_features)
        cls_loss = criterion(logits,source_labels)

        domain_features = torch.cat([source_features,target_features],dim=0)
        domain_labels = torch.cat([torch.zeros(source_features.size(0),dtype=torch.long,device=DEVICE),torch.ones(target_features.size(0),dtype=torch.long,device=DEVICE)])
        domain_logits = discriminator(domain_features,alpha)
        domain_loss = criterion(domain_logits,domain_labels)

        loss = cls_loss+domain_loss
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_domain_loss += domain_loss.item()
        total_domain_acc += (domain_logits.argmax(dim=1)==domain_labels).float().mean().item()
        global_step += 1
    return global_step,total_cls_loss/steps_per_epoch,total_domain_loss/steps_per_epoch,total_domain_acc/steps_per_epoch

def run_training(cfg,max_alpha,checkpoint_name):
    """
    This function runs a complete dann training run for the given
    maximum gradient reversal strength and saves the best checkpoint
    under the given name
    """
    set_seed(cfg["seed"])
    df,splits,train_loaders,val_loaders = load_data(cfg)
    target_loader = load_target_train_loader(df,splits,cfg)
    target_iter = cycle(target_loader)

    backbone,head = build_model()
    discriminator = DomainDiscriminator(backbone.feature_dim,cfg["discriminator"]["hidden_dim"],cfg["discriminator"]["dropout"]).to(DEVICE)
    params = list(backbone.parameters())+list(head.parameters())+list(discriminator.parameters())
    optimizer = torch.optim.AdamW(params,lr=cfg["optimizer"]["lr"],weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(train_loaders[domain]) for domain in SOURCE_DOMAINS)
    total_steps = steps_per_epoch*cfg["optimizer"]["max_epochs"]

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/{checkpoint_name}.pt"

    best_mean_f1 = -1
    epochs_without_improve = 0
    global_step = 0
    history = []

    for epoch in range(cfg["optimizer"]["max_epochs"]):
        global_step,cls_loss,domain_loss,domain_acc = train_epoch(backbone,head,discriminator,train_loaders,target_iter,optimizer,criterion,max_alpha,global_step,total_steps,steps_per_epoch)

        val_f1s = []
        for domain in SOURCE_DOMAINS:
            accuracy,macro_f1 = evaluate(backbone,head,val_loaders[domain])
            val_f1s.append(macro_f1)
            print(f"epoch {epoch} {domain} val_acc {accuracy:.3f} val_f1 {macro_f1:.3f}")
        mean_f1 = sum(val_f1s)/len(val_f1s)
        history.append({"epoch":epoch,"cls_loss":cls_loss,"domain_loss":domain_loss,"domain_acc":domain_acc,"mean_val_f1":mean_f1})

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            save_checkpoint(backbone,head,checkpoint_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg["optimizer"]["patience"]:
            print(f"stopping early at epoch {epoch} best mean_f1 {best_mean_f1:.3f}")
            break

    save_training_curve(history,checkpoint_name,cfg,alignment_key="domain_acc",alignment_label="domain discriminator accuracy")
    return best_mean_f1

def main():
    """
    This function runs the main dann comparison using the fixed maximum
    gradient reversal strength from the config file
    """
    cfg = load_config("dann")
    run_training(cfg,cfg["dann"]["max_alpha"],"dann")

if __name__ == "__main__":
    main()
