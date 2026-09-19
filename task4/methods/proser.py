"""
this file fine tunes proser from the vanilla checkpoint. it appends
randomly initialized dummy classifiers to the closed set head and trains
two complementary objectives on every batch: a classifier placeholder
loss that teaches a dummy class to become the runner up whenever the
true class is excluded, and a data placeholder loss that trains
manifold mixup features blended between two known classes toward that
same dummy group instead of either original class
"""

import os
import torch
import torch.nn.functional as F

from task4.configs.config import load_config,set_seed
from task4.data.cifar10 import build_loader,load_splits,NUM_KNOWN_CLASSES
from task4.methods.common import build_model,save_checkpoint,load_checkpoint,evaluate_accuracy
from task4.methods.manifold_mixup import manifold_mixup
from task4.models.resnet_cifar import extend_classifier
from task4.evaluation.history import save_training_curve
from shared.device import DEVICE

def classifier_placeholder_loss(logits,labels,num_known,beta):
    """
    this function is the proser classifier placeholder objective. an
    ordinary cross entropy term keeps the true known class as the
    strongest response, then the true class is masked out and the
    merged dummy class group is trained to beat every remaining known
    class for the runner up spot
    """
    l1 = F.cross_entropy(logits,labels)

    masked = logits.clone()
    masked[torch.arange(logits.size(0),device=logits.device),labels] = float("-inf")
    known_part = masked[:,:num_known]
    dummy_part = masked[:,num_known:]
    dummy_mass = torch.logsumexp(dummy_part,dim=1,keepdim=True)
    combined = torch.cat([known_part,dummy_mass],dim=1)
    dummy_target = torch.full((logits.size(0),),num_known,dtype=torch.long,device=logits.device)
    l2 = F.cross_entropy(combined,dummy_target)

    return l1+beta*l2

def data_placeholder_loss(mixed_logits,num_known):
    """
    this function trains a manifold mixup feature toward the merged
    dummy class group instead of either of the two known classes it was
    blended from
    """
    dummy_mass = torch.logsumexp(mixed_logits[:,num_known:],dim=1,keepdim=True)
    combined = torch.cat([mixed_logits[:,:num_known],dummy_mass],dim=1)
    target = torch.full((mixed_logits.size(0),),num_known,dtype=torch.long,device=mixed_logits.device)
    return F.cross_entropy(combined,target)

def train_one_epoch(model,loader,optimizer,num_known,beta,gamma,mixup_alpha):
    """
    this function runs one training epoch. it splits every batch in half,
    trains the first half with the classifier placeholder loss and the
    second half with the manifold mixup data placeholder loss, and
    returns the average of each loss for the epoch
    """
    model.train()
    total_cls = 0.0
    total_data = 0.0
    n_batches = 0
    for images,labels in loader:
        images,labels = images.to(DEVICE),labels.to(DEVICE)
        half = images.size(0)//2
        cls_images,cls_labels = images[:half],labels[:half]
        mix_images,mix_labels = images[half:],labels[half:]

        optimizer.zero_grad()

        cls_logits,_ = model(cls_images)
        cls_loss = classifier_placeholder_loss(cls_logits,cls_labels,num_known,beta)

        mixed_pre = manifold_mixup(model,mix_images,mix_labels,mixup_alpha)
        mixed_logits,_ = model.forward_post(mixed_pre)
        data_loss = data_placeholder_loss(mixed_logits,num_known)

        loss = cls_loss+gamma*data_loss
        loss.backward()
        optimizer.step()

        total_cls += cls_loss.item()
        total_data += data_loss.item()
        n_batches += 1
    return total_cls/n_batches,total_data/n_batches

def run_training(cfg):
    """
    this function fine tunes proser from the vanilla checkpoint for the
    configured number of epochs and keeps the checkpoint with the
    highest cifar ten validation accuracy over the ten known classes
    """
    set_seed(cfg["seed"])
    splits = load_splits(cfg)
    train_loader = build_loader(cfg,splits["train"],augment=True,batch_size=cfg["data"]["batch_size"],shuffle=True)
    val_loader = build_loader(cfg,splits["val"],augment=False,batch_size=cfg["data"]["eval_batch_size"],shuffle=False)

    proser_cfg = cfg["proser"]
    model = build_model(NUM_KNOWN_CLASSES)
    vanilla_checkpoint = f"{cfg['paths']['checkpoints_dir']}/vanilla.pt"
    load_checkpoint(model,vanilla_checkpoint)
    extend_classifier(model,proser_cfg["num_dummy_classes"])

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.SGD(model.parameters(),lr=opt_cfg["lr"],momentum=opt_cfg["momentum"],weight_decay=opt_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt_cfg["max_epochs"])

    os.makedirs(cfg["paths"]["checkpoints_dir"],exist_ok=True)
    checkpoint_path = f"{cfg['paths']['checkpoints_dir']}/proser.pt"

    best_val_acc = -1
    history = []
    for epoch in range(opt_cfg["max_epochs"]):
        cls_loss,data_loss = train_one_epoch(model,train_loader,optimizer,NUM_KNOWN_CLASSES,proser_cfg["beta"],proser_cfg["gamma"],proser_cfg["mixup_alpha"])
        val_acc = evaluate_accuracy(model,val_loader)
        scheduler.step()
        print(f"epoch {epoch} cls_loss {cls_loss:.4f} data_loss {data_loss:.4f} val_acc {val_acc:.3f}")
        history.append({"epoch":epoch,"loss":cls_loss,"data_loss":data_loss,"val_acc":val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model,checkpoint_path)

    save_training_curve(history,"proser",cfg,alignment_key="data_loss",alignment_label="data placeholder loss")
    print(f"best proser val_acc {best_val_acc:.3f}")
    return history

def main():
    """
    this function runs the main proser fine tuning run using the
    settings from the config file
    """
    cfg = load_config("proser")
    run_training(cfg)

if __name__ == "__main__":
    main()
