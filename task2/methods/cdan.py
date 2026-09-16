"""
This file trains the resnet eighteen backbone and its linear head using
cdan style class conditional adversarial alignment. The domain
discriminator sees the outer product of the backbone feature and the
predicted class probability instead of the feature alone
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.configs.config import load_config,set_seed
from task2.methods.common import load_data,load_target_train_loader,build_model,save_checkpoint
from task2.evaluation.metrics import evaluate
from task2.models.domain_discriminator import DomainDiscriminator,grl_alpha
from shared.bn_utils import freeze_batchnorm
from shared.pacs_protocol import SOURCE_DOMAINS,cycle

def condition_on_prediction(features,probs):
    """
    This function builds the conditioning vector cdan feeds to its
    domain discriminator by taking the outer product of every feature
    vector with its predicted class probabilities and flattening the
    result back into one vector per example
    """
    batch = features.size(0)
    outer = features.unsqueeze(2)*probs.unsqueeze(1)
    return outer.view(batch,-1)

def train_epoch(backbone,head,discriminator,train_loaders,target_iter,optimizer,criterion,max_alpha,global_step,total_steps,steps_per_epoch):
    """
    This function runs one epoch of cdan training and returns the
    updated global step count so the gradient reversal schedule keeps
    advancing across later epochs
    """
    backbone.train()
    head.train()
    discriminator.train()
    freeze_batchnorm(backbone)
    iterators = {domain:cycle(train_loaders[domain]) for domain in SOURCE_DOMAINS}
    for _ in range(steps_per_epoch):
        images = []
        labels = []
        for domain in SOURCE_DOMAINS:
            batch_images,batch_labels,_ = next(iterators[domain])
            images.append(batch_images)
            labels.append(batch_labels)
        source_images = torch.cat(images,dim=0)
        source_labels = torch.cat(labels,dim=0)
        target_images,_,_ = next(target_iter)

        progress = global_step/total_steps
        alpha = grl_alpha(progress,max_alpha)

        optimizer.zero_grad()
        source_features = backbone(source_images)
        target_features = backbone(target_images)
        source_logits = head(source_features)
        target_logits = head(target_features)
        cls_loss = criterion(source_logits,source_labels)

        source_probs = F.softmax(source_logits,dim=1)
        target_probs = F.softmax(target_logits,dim=1)
        source_conditioned = condition_on_prediction(source_features,source_probs)
        target_conditioned = condition_on_prediction(target_features,target_probs)

        domain_features = torch.cat([source_conditioned,target_conditioned],dim=0)
        domain_labels = torch.cat([torch.zeros(source_features.size(0),dtype=torch.long),torch.ones(target_features.size(0),dtype=torch.long)])
        domain_logits = discriminator(domain_features,alpha)
        domain_loss = criterion(domain_logits,domain_labels)

        loss = cls_loss+domain_loss
        loss.backward()
        optimizer.step()

        global_step += 1
    return global_step

def run_training(cfg,max_alpha,checkpoint_name):
    """
    This function runs a complete cdan training run for the given
    maximum gradient reversal strength and saves the best checkpoint
    under the given name
    """
    set_seed(cfg["seed"])
    df,splits,train_loaders,val_loaders = load_data(cfg)
    target_loader = load_target_train_loader(df,splits,cfg)
    target_iter = cycle(target_loader)

    backbone,head = build_model()
    num_classes = head.linear.out_features
    discriminator = DomainDiscriminator(backbone.feature_dim*num_classes,cfg["discriminator"]["hidden_dim"],cfg["discriminator"]["dropout"])
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

    for epoch in range(cfg["optimizer"]["max_epochs"]):
        global_step = train_epoch(backbone,head,discriminator,train_loaders,target_iter,optimizer,criterion,max_alpha,global_step,total_steps,steps_per_epoch)

        val_f1s = []
        for domain in SOURCE_DOMAINS:
            accuracy,macro_f1 = evaluate(backbone,head,val_loaders[domain])
            val_f1s.append(macro_f1)
            print(f"epoch {epoch} {domain} val_acc {accuracy:.3f} val_f1 {macro_f1:.3f}")
        mean_f1 = sum(val_f1s)/len(val_f1s)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            save_checkpoint(backbone,head,checkpoint_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg["optimizer"]["patience"]:
            print(f"stopping early at epoch {epoch} best mean_f1 {best_mean_f1:.3f}")
            break

    return best_mean_f1

def main():
    """
    This function runs the main cdan comparison using the fixed maximum
    gradient reversal strength from the config file
    """
    cfg = load_config("cdan")
    run_training(cfg,cfg["cdan"]["max_alpha"],"cdan")

if __name__ == "__main__":
    main()
