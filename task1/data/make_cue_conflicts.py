"""
This file builds the shape versus texture cue conflict images used in
task one. A content image from one class is combined with the texture
of a style image from a different class using the pretrained adain
network. A simple visual rule then rejects any result where the
original shape is no longer visible
"""

import json
import os
import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

from configs.config import load_config,set_seed
from data.dataset import load_splits,load_stl10,images_to_array
from models.adain import AdainStyleTransfer

def group_indices_by_class(labels):
    """
    This function groups the position of every label by its class id
    so images belonging to one class can be looked up quickly
    """
    groups = {}
    for position,label in enumerate(labels):
        class_id = int(label)
        if class_id not in groups:
            groups[class_id] = []
        groups[class_id].append(position)
    return groups

def is_shape_preserved(content_img,stylized_img,threshold):
    """
    This function checks whether the stylized image still looks like
    the content image in terms of structure. It compares both images
    in grayscale using structural similarity and rejects the pair if
    the score falls below the given threshold
    """
    content_gray = rgb2gray(content_img)
    stylized_gray = rgb2gray(stylized_img)
    score = structural_similarity(content_gray,stylized_gray,data_range=1.0)
    return score >= threshold

def main():
    """
    This function generates the required number of cue conflict images
    for every class pair and both directions then saves the images
    that pass the rejection rule together with their labels
    """
    cfg = load_config()
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    splits = load_splits(cfg)
    classes = splits["classes"]
    class_to_id = {name:i for i,name in enumerate(classes)}

    _,test_ds = load_stl10(cfg)
    eval_images,eval_labels = images_to_array(test_ds,splits["eval_idx"])
    groups = group_indices_by_class(eval_labels)

    net = AdainStyleTransfer()
    net.eval()

    cue_cfg = cfg["cue_conflict"]
    alpha = cue_cfg["style_alpha"]
    n_per_combo = cue_cfg["images_per_pair_direction"]
    threshold = cue_cfg["rejection_ssim_threshold"]

    kept_images = []
    kept_meta = []
    n_rejected = 0

    for class_a,class_b in cue_cfg["class_pairs"]:
        for content_name,style_name in [(class_a,class_b),(class_b,class_a)]:
            content_id = class_to_id[content_name]
            style_id = class_to_id[style_name]
            content_pool = groups[content_id]
            style_pool = groups[style_id]

            content_choice = rng.choice(content_pool,size=n_per_combo,replace=False)
            style_choice = rng.choice(style_pool,size=n_per_combo,replace=False)

            content_batch = torch.from_numpy(eval_images[content_choice]).permute(0,3,1,2).float()
            style_batch = torch.from_numpy(eval_images[style_choice]).permute(0,3,1,2).float()

            stylized = net(content_batch,style_batch,alpha=alpha)
            stylized = stylized.permute(0,2,3,1).numpy()

            for i in range(n_per_combo):
                content_img = eval_images[content_choice[i]]
                stylized_img = stylized[i]
                if is_shape_preserved(content_img,stylized_img,threshold):
                    kept_images.append(stylized_img)
                    kept_meta.append({
                        "content_class":content_name,
                        "style_class":style_name,
                        "content_eval_index":int(content_choice[i]),
                        "style_eval_index":int(style_choice[i]),
                    })
                else:
                    n_rejected += 1

    kept_images = np.stack(kept_images,axis=0)
    kept_images_uint8 = (kept_images*255.0).round().astype(np.uint8)

    images_dir = cfg["paths"]["cache_dir"]+"/images"
    os.makedirs(images_dir,exist_ok=True)
    np.save(images_dir+"/cue_conflict.npy",kept_images_uint8)

    meta = {
        "items":kept_meta,
        "n_rejected":n_rejected,
        "n_kept":len(kept_meta),
    }
    with open(cfg["paths"]["cache_dir"]+"/cue_conflict_meta.json","w") as f:
        json.dump(meta,f,indent=2)

    print(f"kept {len(kept_meta)} cue conflicts and rejected {n_rejected}")
    if len(kept_meta) < cue_cfg["min_valid_conflicts"]:
        print(f"warning the kept count is below the required minimum of {cue_cfg['min_valid_conflicts']}")

if __name__ == "__main__":
    main()
