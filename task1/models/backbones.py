"""
This file wraps the three pretrained backbones used in this task so
every other script can call them in the same way. Each backbone is
frozen and returns the exact feature vector that the assignment asks
for. A resnet gives its pooled feature. A vit gives its class token
Clip gives its normalized image embedding
"""

import torch
import torch.nn as nn
import torchvision
import open_clip

IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD = [0.229,0.224,0.225]
CLIP_MEAN = [0.48145466,0.4578275,0.40821073]
CLIP_STD = [0.26862954,0.26130258,0.27577711]

def normalize_batch(images,mean,std):
    """
    This function takes a batch of images already scaled between zero
    and one and applies the mean and the standard deviation that a
    particular backbone expects
    """
    device = images.device
    mean_t = torch.tensor(mean,device=device).view(1,3,1,1)
    std_t = torch.tensor(std,device=device).view(1,3,1,1)
    return (images-mean_t)/std_t

class ResnetBackbone(nn.Module):
    """
    This class wraps torchvision resnet fifty pretrained on imagenet
    and exposes its pooled feature as the output instead of the class
    scores it was originally trained with
    """

    feature_dim = 2048
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Identity()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self,images):
        """
        This method expects a batch of images already resized to two
        hundred twenty four pixels and scaled between zero and one. It
        returns the pooled feature for every image in the batch
        """
        x = normalize_batch(images,self.mean,self.std)
        return self.model(x)

class VitBackbone(nn.Module):
    """
    This class wraps torchvision vit base sixteen pretrained on
    imagenet and exposes its class token as the output instead of the
    class scores it was originally trained with
    """

    feature_dim = 768
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = torchvision.models.vit_b_16(weights=weights)
        model.heads = nn.Identity()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self,images):
        """
        This method expects a batch of images already resized to two
        hundred twenty four pixels and scaled between zero and one. It
        returns the final class token for every image in the batch
        """
        x = normalize_batch(images,self.mean,self.std)
        return self.model(x)

class ClipBackbone(nn.Module):
    """
    This class wraps the open clip vit base thirty two model with the
    original openai weights. The quick gelu version of the model is
    used on purpose because that matches the activation function the
    openai weights were actually trained with
    """

    feature_dim = 512
    mean = CLIP_MEAN
    std = CLIP_STD

    def __init__(self):
        super().__init__()
        model,_,_ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu",pretrained="openai")
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    def forward(self,images):
        """
        This method expects a batch of images already resized to two
        hundred twenty four pixels and scaled between zero and one. It
        returns the image embedding after it has been scaled to unit
        length as required by the assignment
        """
        x = normalize_batch(images,self.mean,self.std)
        feat = self.model.encode_image(x)
        feat = feat/feat.norm(dim=-1,keepdim=True)
        return feat

    def zero_shot_logits(self,images,class_prompts):
        """
        This method compares every image embedding against every class
        prompt embedding and returns the similarity scores used for
        zero shot classification. The scores are already multiplied by
        the learned temperature so a softmax over them gives the same
        confidence clip itself reports
        """
        image_feat = self.forward(images)
        tokens = self.tokenizer(class_prompts)
        text_feat = self.model.encode_text(tokens)
        text_feat = text_feat/text_feat.norm(dim=-1,keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale*image_feat@text_feat.t()

def extract_features(backbone,images,batch_size=64):
    """
    This function runs a backbone over a large set of images a small
    batch at a time and stacks the results together. Working in small
    batches keeps memory use low since no gradients are needed here
    """
    backbone.eval()
    all_feats = []
    with torch.no_grad():
        for start in range(0,len(images),batch_size):
            chunk = images[start:start+batch_size]
            feats = backbone(chunk)
            all_feats.append(feats)
    return torch.cat(all_feats,dim=0)
