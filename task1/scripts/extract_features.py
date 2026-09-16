"""
This file is where every backbone actually looks at an image. For
every condition it builds the transformed images in memory extracts
the feature for all three backbones right away and then throws the
pixel array away. Doing it this way means the many transformed
versions of the eval set never have to be saved to disk since only
the small feature vectors need to survive between scripts
"""

import json
import os
import numpy as np
import torch

from configs.config import load_config,set_seed
from data.dataset import load_splits,load_stl10,images_to_array
from data.transforms import (
    build_color_stats,
    build_grayscale_batch,
    build_color_swap_batch,
    build_translation_variant,
    build_patch_shuffle_batch,
)
from models.backbones import ResnetBackbone,VitBackbone,ClipBackbone,extract_features

def images_to_tensor(images):
    """
    This function turns a numpy array of images shaped as images
    height width channel into a torch tensor shaped as images channel
    height width which is what every backbone expects
    """
    return torch.from_numpy(images).permute(0,3,1,2).float()

def extract_condition(backbones,images,condition_name,features_dir):
    """
    This function takes one array of images and runs every backbone on
    it once then saves the resulting features. The same image array is
    reused for every backbone so all three see the exact same pixels
    """
    tensor_images = images_to_tensor(images)
    for backbone_name,backbone in backbones.items():
        feats = extract_features(backbone,tensor_images)
        out_path = f"{features_dir}/{backbone_name}_{condition_name}.pt"
        torch.save(feats,out_path)
    print(f"extracted features for condition {condition_name}")

def main():
    """
    This function builds every required condition one at a time and
    extracts and saves the features for it before moving on to the
    next condition so pixel arrays never pile up on disk
    """
    cfg = load_config()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    splits = load_splits(cfg)
    train_ds,test_ds = load_stl10(cfg)

    features_dir = cfg["paths"]["cache_dir"]+"/features"
    os.makedirs(features_dir,exist_ok=True)

    backbones = {
        "resnet":ResnetBackbone(),
        "vit":VitBackbone(),
        "clip":ClipBackbone(),
    }

    train_images,train_labels = images_to_array(train_ds,splits["train_idx"])
    extract_condition(backbones,train_images,"train",features_dir)

    val_images,val_labels = images_to_array(train_ds,splits["val_idx"])
    extract_condition(backbones,val_images,"val",features_dir)

    eval_images,eval_labels = images_to_array(test_ds,splits["eval_idx"])
    extract_condition(backbones,eval_images,"eval_clean",features_dir)

    color_stats = build_color_stats(train_images,train_labels)
    with open(cfg["paths"]["cache_dir"]+"/color_stats.json","w") as f:
        json.dump(color_stats,f,indent=2)

    grayscale_images = build_grayscale_batch(eval_images)
    extract_condition(backbones,grayscale_images,"eval_grayscale",features_dir)
    del grayscale_images

    colorswap_images = build_color_swap_batch(eval_images,eval_labels,color_stats,rng)
    extract_condition(backbones,colorswap_images,"eval_colorswap",features_dir)
    del colorswap_images

    grid_size = cfg["interventions"]["patch_shuffle"]["grid_size"]
    patch_images = build_patch_shuffle_batch(eval_images,grid_size,rng)
    extract_condition(backbones,patch_images,"eval_patchshuffle",features_dir)
    del patch_images

    translation_cfg = cfg["interventions"]["translation"]
    displacements = translation_cfg["displacements"]
    directions = translation_cfg["directions"]
    max_disp = max(displacements)

    extract_condition(backbones,eval_images,"eval_translate_d0",features_dir)
    for d in displacements:
        if d == 0:
            continue
        for direction in directions:
            variant = build_translation_variant(eval_images,d,direction,max_disp)
            extract_condition(backbones,variant,f"eval_translate_d{d}_{direction}",features_dir)
            del variant

    cue_conflict_path = cfg["paths"]["cache_dir"]+"/images/cue_conflict.npy"
    if os.path.exists(cue_conflict_path):
        cue_images = np.load(cue_conflict_path).astype(np.float32)/255.0
        extract_condition(backbones,cue_images,"cue_conflict",features_dir)

    np.savez(features_dir+"/labels.npz",train=train_labels,val=val_labels,eval_labels=eval_labels)
    print("saved all labels")

if __name__ == "__main__":
    main()
