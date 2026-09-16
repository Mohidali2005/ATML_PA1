"""
This file compares the clean baseline against every intervention for
all three backbones. It works out accuracy macro f1 mean confidence
and prediction consistency and it also works out the shape versus
texture cue conflict numbers
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from configs.config import load_config
from models.backbones import ClipBackbone

def load_head(path):
    """
    This function loads a saved linear head back from disk. The shape
    of its saved weight already tells us how many input features and
    how many classes it has
    """
    state = torch.load(path)
    num_classes,feature_dim = state["weight"].shape
    head = nn.Linear(feature_dim,num_classes)
    head.load_state_dict(state)
    head.eval()
    return head

def predict_with_head(head,feats):
    """
    This function runs the given features through a linear head and
    returns the predicted class and the confidence of that prediction
    for every image
    """
    with torch.no_grad():
        logits = head(feats)
        probs = torch.softmax(logits,dim=1)
        confidence,prediction = probs.max(dim=1)
    return prediction.numpy(),confidence.numpy()

def predict_zero_shot(clip_backbone,image_feats,text_feats):
    """
    This function scores cached clip image features against the class
    text features and returns the predicted class and the confidence
    for every image
    """
    logit_scale = clip_backbone.model.logit_scale.exp()
    with torch.no_grad():
        logits = logit_scale*image_feats@text_feats.t()
        probs = torch.softmax(logits,dim=1)
        confidence,prediction = probs.max(dim=1)
    return prediction.numpy(),confidence.numpy()

def summarize(labels,prediction,confidence):
    """
    This function turns raw predictions into the three summary numbers
    used everywhere in this task which are top one accuracy macro f1
    and mean confidence
    """
    accuracy = float((prediction==labels).mean())
    macro_f1 = float(f1_score(labels,prediction,average="macro"))
    mean_confidence = float(confidence.mean())
    return accuracy,macro_f1,mean_confidence

def consistency(prediction_a,prediction_b):
    """
    This function returns the fraction of images whose predicted class
    did not change between two conditions
    """
    return float((prediction_a==prediction_b).mean())

def get_text_features(clip_backbone,classes,prompt_template):
    """
    This function builds one text prompt per class using the fixed
    template from the config and returns their normalized clip text
    features
    """
    prompts = [prompt_template.format(**{"class":name}) for name in classes]
    tokens = clip_backbone.tokenizer(prompts)
    with torch.no_grad():
        text_feats = clip_backbone.model.encode_text(tokens)
        text_feats = text_feats/text_feats.norm(dim=-1,keepdim=True)
    return text_feats

def get_predictions(backbone_name,condition,heads,clip_backbone,text_feats,features_dir):
    """
    This function loads the cached features for one backbone and one
    condition and returns the prediction and the confidence for every
    image. Clip also gets a zero shot prediction in addition to its
    trained head prediction
    """
    feats = torch.load(f"{features_dir}/{backbone_name}_{condition}.pt")
    prediction,confidence = predict_with_head(heads[backbone_name],feats)
    result = {"head":(prediction,confidence)}
    if backbone_name == "clip":
        result["zero_shot"] = predict_zero_shot(clip_backbone,feats,text_feats)
    return result

def build_clean_table(backbones,heads,clip_backbone,text_feats,features_dir,eval_labels):
    """
    This function builds the clean baseline table with one row per
    scoring method which is one row for every trained head plus one
    extra row for clip zero shot
    """
    rows = []
    for backbone_name in backbones:
        predictions = get_predictions(backbone_name,"eval_clean",heads,clip_backbone,text_feats,features_dir)
        prediction,confidence = predictions["head"]
        accuracy,macro_f1,mean_confidence = summarize(eval_labels,prediction,confidence)
        rows.append({
            "method":f"{backbone_name}_head",
            "accuracy":accuracy,
            "macro_f1":macro_f1,
            "mean_confidence":mean_confidence,
        })
        if "zero_shot" in predictions:
            prediction,confidence = predictions["zero_shot"]
            accuracy,macro_f1,mean_confidence = summarize(eval_labels,prediction,confidence)
            rows.append({
                "method":"clip_zero_shot",
                "accuracy":accuracy,
                "macro_f1":macro_f1,
                "mean_confidence":mean_confidence,
            })
    return pd.DataFrame(rows)

def build_condition_table(backbones,heads,clip_backbone,text_feats,features_dir,eval_labels,conditions):
    """
    This function builds one row per backbone per condition with its
    accuracy macro f1 and consistency compared with the clean
    prediction from the same backbone head
    """
    rows = []
    for backbone_name in backbones:
        clean_prediction,_ = get_predictions(backbone_name,"eval_clean",heads,clip_backbone,text_feats,features_dir)["head"]
        for condition in conditions:
            prediction,confidence = get_predictions(backbone_name,condition,heads,clip_backbone,text_feats,features_dir)["head"]
            accuracy,macro_f1,mean_confidence = summarize(eval_labels,prediction,confidence)
            rows.append({
                "backbone":backbone_name,
                "condition":condition,
                "accuracy":accuracy,
                "macro_f1":macro_f1,
                "mean_confidence":mean_confidence,
                "consistency":consistency(prediction,clean_prediction),
            })
    return pd.DataFrame(rows)

def build_translation_table(backbones,heads,clip_backbone,text_feats,features_dir,eval_labels,displacements,directions):
    """
    This function averages accuracy and consistency over the four
    directions for every displacement so the result can be plotted as
    one curve per backbone
    """
    rows = []
    for backbone_name in backbones:
        clean_prediction,_ = get_predictions(backbone_name,"eval_clean",heads,clip_backbone,text_feats,features_dir)["head"]
        for d in displacements:
            if d == 0:
                condition_keys = ["eval_translate_d0"]
            else:
                condition_keys = [f"eval_translate_d{d}_{direction}" for direction in directions]

            accuracies = []
            consistencies = []
            for condition in condition_keys:
                prediction,_ = get_predictions(backbone_name,condition,heads,clip_backbone,text_feats,features_dir)["head"]
                accuracies.append(float((prediction==eval_labels).mean()))
                consistencies.append(consistency(prediction,clean_prediction))

            rows.append({
                "backbone":backbone_name,
                "displacement":d,
                "accuracy":float(np.mean(accuracies)),
                "consistency":float(np.mean(consistencies)),
            })
    return pd.DataFrame(rows)

def build_cue_conflict_table(backbones,heads,clip_backbone,text_feats,features_dir,classes,cue_meta):
    """
    This function scores every cue conflict image and counts how often
    the prediction matches the content class the style class or
    neither. It reports these counts together with the shape bias
    percentage and the coverage percentage from the assignment
    """
    class_to_id = {name:i for i,name in enumerate(classes)}
    content_ids = np.array([class_to_id[item["content_class"]] for item in cue_meta["items"]])
    style_ids = np.array([class_to_id[item["style_class"]] for item in cue_meta["items"]])

    rows = []
    for backbone_name in backbones:
        predictions = get_predictions(backbone_name,"cue_conflict",heads,clip_backbone,text_feats,features_dir)
        methods = {"head":predictions["head"]}
        if "zero_shot" in predictions:
            methods["zero_shot"] = predictions["zero_shot"]

        for method_name,(prediction,_) in methods.items():
            n_shape = int((prediction==content_ids).sum())
            n_texture = int((prediction==style_ids).sum())
            n_total = len(prediction)
            n_other = n_total-n_shape-n_texture
            shape_bias = 100.0*n_shape/max(n_shape+n_texture,1)
            coverage = 100.0*(n_shape+n_texture)/n_total
            rows.append({
                "backbone":backbone_name,
                "method":method_name,
                "n_shape":n_shape,
                "n_texture":n_texture,
                "n_other":n_other,
                "shape_bias_percent":shape_bias,
                "coverage_percent":coverage,
            })
    return pd.DataFrame(rows)

def save_cue_conflict_examples(backbones,heads,clip_backbone,text_feats,features_dir,classes,cue_meta,cache_dir,figures_dir):
    """
    This function picks a small set of cue conflict images that show
    interesting behavior and saves them as one figure. Some examples
    are cases where every method agrees on the shape some are cases
    where every method agrees on the texture and some are cases where
    the methods disagree with each other
    """
    images = np.load(cache_dir+"/images/cue_conflict.npy").astype(np.float32)/255.0

    method_predictions = {}
    for backbone_name in backbones:
        predictions = get_predictions(backbone_name,"cue_conflict",heads,clip_backbone,text_feats,features_dir)
        method_predictions[f"{backbone_name}_head"] = predictions["head"][0]
        if "zero_shot" in predictions:
            method_predictions["clip_zero_shot"] = predictions["zero_shot"][0]

    method_names = list(method_predictions.keys())
    n_items = len(cue_meta["items"])

    agree_shape = []
    agree_texture = []
    disagree = []

    for i in range(n_items):
        content_name = cue_meta["items"][i]["content_class"]
        style_name = cue_meta["items"][i]["style_class"]
        content_id = classes.index(content_name)
        style_id = classes.index(style_name)
        votes = [method_predictions[name][i] for name in method_names]

        if all(vote==content_id for vote in votes):
            agree_shape.append(i)
        elif all(vote==style_id for vote in votes):
            agree_texture.append(i)
        elif len(set(votes)) > 1:
            disagree.append(i)

    chosen = agree_shape[:3]+agree_texture[:3]+disagree[:3]
    chosen = chosen[:9]

    fig,axes = plt.subplots(3,3,figsize=(11,11))
    for ax,item_idx in zip(axes.flat,chosen):
        ax.imshow(images[item_idx])
        content_name = cue_meta["items"][item_idx]["content_class"]
        style_name = cue_meta["items"][item_idx]["style_class"]
        votes_text = " ".join(f"{name}={classes[method_predictions[name][item_idx]]}" for name in method_names)
        ax.set_title(f"shape {content_name} texture {style_name}\n{votes_text}",fontsize=7)
        ax.axis("off")

    for ax in axes.flat[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = figures_dir+"/cue_conflict_examples.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")

def plot_intervention_comparison(clean_table,color_table,patch_table,figures_dir):
    """
    This function draws one bar chart that puts the clean grayscale
    color swapped and patch shuffled accuracy for every backbone next
    to each other so they are easy to compare at a glance
    """
    backbone_colors = {"resnet":"steelblue","vit":"darkorange","clip":"seagreen"}
    condition_order = ["clean","eval_grayscale","eval_colorswap","eval_patchshuffle"]
    condition_labels = {
        "clean":"clean",
        "eval_grayscale":"grayscale",
        "eval_colorswap":"color swap",
        "eval_patchshuffle":"patch shuffle",
    }

    rows = []
    for backbone_name in backbone_colors:
        clean_accuracy = clean_table.loc[clean_table["method"]==f"{backbone_name}_head","accuracy"].iloc[0]
        rows.append({"backbone":backbone_name,"condition":"clean","accuracy":clean_accuracy})
    for _,row in color_table.iterrows():
        rows.append({"backbone":row["backbone"],"condition":row["condition"],"accuracy":row["accuracy"]})
    for _,row in patch_table.iterrows():
        rows.append({"backbone":row["backbone"],"condition":row["condition"],"accuracy":row["accuracy"]})
    combined = pd.DataFrame(rows)

    fig,ax = plt.subplots(figsize=(8,5))
    bar_width = 0.25
    x = np.arange(len(condition_order))
    for i,backbone_name in enumerate(backbone_colors):
        values = [
            combined[(combined["backbone"]==backbone_name)&(combined["condition"]==c)]["accuracy"].iloc[0]
            for c in condition_order
        ]
        ax.bar(x+i*bar_width,values,width=bar_width,label=backbone_name,color=backbone_colors[backbone_name])

    ax.set_xticks(x+bar_width)
    ax.set_xticklabels([condition_labels[c] for c in condition_order])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0,1.05)
    ax.legend()
    plt.tight_layout()

    out_path = figures_dir+"/intervention_comparison.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")

def plot_translation_curve(translation_table,figures_dir):
    """
    This function draws the translation curve which is accuracy and
    consistency plotted against the pixel displacement for every
    backbone
    """
    backbone_colors = {"resnet":"steelblue","vit":"darkorange","clip":"seagreen"}

    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for backbone_name,color in backbone_colors.items():
        subset = translation_table[translation_table["backbone"]==backbone_name].sort_values("displacement")
        axes[0].plot(subset["displacement"],subset["accuracy"],label=backbone_name,color=color,marker="o")
        axes[1].plot(subset["displacement"],subset["consistency"],label=backbone_name,color=color,marker="o")

    axes[0].set_xlabel("displacement in pixels")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()

    axes[1].set_xlabel("displacement in pixels")
    axes[1].set_ylabel("consistency with the clean prediction")
    axes[1].legend()

    plt.tight_layout()
    out_path = figures_dir+"/translation_curve.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")

def main():
    """
    This function loads every trained head and every cached feature
    set then builds and saves all the comparison tables required by
    the assignment
    """
    cfg = load_config()
    features_dir = cfg["paths"]["cache_dir"]+"/features"
    heads_dir = cfg["paths"]["cache_dir"]+"/heads"
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(tables_dir,exist_ok=True)
    os.makedirs(figures_dir,exist_ok=True)

    splits_path = cfg["paths"]["cache_dir"]+"/splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    classes = splits["classes"]

    labels = np.load(features_dir+"/labels.npz")
    eval_labels = labels["eval_labels"]

    backbones = ["resnet","vit","clip"]
    heads = {name:load_head(f"{heads_dir}/{name}_head.pt") for name in backbones}

    clip_backbone = ClipBackbone()
    text_feats = get_text_features(clip_backbone,classes,cfg["clip"]["zero_shot_prompt"])

    clean_table = build_clean_table(backbones,heads,clip_backbone,text_feats,features_dir,eval_labels)
    clean_table.to_csv(f"{tables_dir}/clean_baseline.csv",index=False)
    print(clean_table)

    color_conditions = ["eval_grayscale","eval_colorswap"]
    color_table = build_condition_table(backbones,heads,clip_backbone,text_feats,features_dir,eval_labels,color_conditions)
    color_table.to_csv(f"{tables_dir}/color_bias.csv",index=False)
    print(color_table)

    patch_table = build_condition_table(backbones,heads,clip_backbone,text_feats,features_dir,eval_labels,["eval_patchshuffle"])
    patch_table.to_csv(f"{tables_dir}/patch_structure.csv",index=False)
    print(patch_table)

    plot_intervention_comparison(clean_table,color_table,patch_table,figures_dir)

    translation_cfg = cfg["interventions"]["translation"]
    translation_table = build_translation_table(
        backbones,heads,clip_backbone,text_feats,features_dir,eval_labels,
        translation_cfg["displacements"],translation_cfg["directions"],
    )
    translation_table.to_csv(f"{tables_dir}/translation_curve.csv",index=False)
    print(translation_table)

    plot_translation_curve(translation_table,figures_dir)

    with open(cfg["paths"]["cache_dir"]+"/cue_conflict_meta.json") as f:
        cue_meta = json.load(f)
    cue_table = build_cue_conflict_table(backbones,heads,clip_backbone,text_feats,features_dir,classes,cue_meta)
    cue_table.to_csv(f"{tables_dir}/cue_conflict.csv",index=False)
    print(cue_table)

    save_cue_conflict_examples(
        backbones,heads,clip_backbone,text_feats,features_dir,classes,cue_meta,
        cfg["paths"]["cache_dir"],figures_dir,
    )

if __name__ == "__main__":
    main()
