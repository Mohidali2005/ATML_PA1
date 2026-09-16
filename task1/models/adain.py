"""
This file rebuilds the AdaIN style transfer network from the paper by
Huang and Belongie. The encoder and decoder weights are pretrained and
are only loaded from disk here. Nothing in this file is trained again
"""

import pathlib
import torch
import torch.nn as nn

WEIGHTS_DIR = pathlib.Path(__file__).resolve().parent/"pretrained"

decoder = nn.Sequential(
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,256,(3,3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2,mode="nearest"),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,256,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,256,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,256,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,128,(3,3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2,mode="nearest"),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(128,128,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(128,64,(3,3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2,mode="nearest"),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(64,64,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(64,3,(3,3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3,3,(1,1)),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(3,64,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(64,64,(3,3)),
    nn.ReLU(),
    nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(64,128,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(128,128,(3,3)),
    nn.ReLU(),
    nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(128,256,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,256,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,256,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,256,(3,3)),
    nn.ReLU(),
    nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(256,512,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
    nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1,1,1,1)),
    nn.Conv2d(512,512,(3,3)),
    nn.ReLU(),
)

def calc_mean_std(feat,eps=1e-5):
    """
    This function returns the mean and the standard deviation of a
    feature map computed separately for every image and every channel
    A small value is added before the square root so it never divides
    by zero
    """
    n,c = feat.shape[0],feat.shape[1]
    feat_var = feat.view(n,c,-1).var(dim=2)+eps
    feat_std = feat_var.sqrt().view(n,c,1,1)
    feat_mean = feat.view(n,c,-1).mean(dim=2).view(n,c,1,1)
    return feat_mean,feat_std

def adaptive_instance_normalization(content_feat,style_feat):
    """
    This function normalizes the content feature map and then rescales
    it using the mean and the standard deviation of the style feature
    map. The spatial layout of the content stays the same while its
    channel statistics become those of the style
    """
    size = content_feat.size()
    style_mean,style_std = calc_mean_std(style_feat)
    content_mean,content_std = calc_mean_std(content_feat)
    normalized = (content_feat-content_mean.expand(size))/content_std.expand(size)
    return normalized*style_std.expand(size)+style_mean.expand(size)

class AdainStyleTransfer(nn.Module):
    """
    This class wraps the pretrained encoder and decoder into a single
    module. Calling it on a content image and a style image returns a
    new image that keeps the shape of the content image but takes on
    the texture and color of the style image
    """

    def __init__(self):
        """
        This method loads the pretrained vgg encoder and the pretrained
        decoder from the pretrained folder next to this file and freezes
        every parameter since nothing here is trained again
        """
        super().__init__()
        vgg_net = vgg
        vgg_net.load_state_dict(torch.load(WEIGHTS_DIR/"vgg_normalised.pth",map_location="cpu"))
        vgg_net = nn.Sequential(*list(vgg_net.children())[:31])

        dec_net = decoder
        dec_net.load_state_dict(torch.load(WEIGHTS_DIR/"decoder.pth",map_location="cpu"))

        enc_layers = list(vgg_net.children())
        self.enc_1 = nn.Sequential(*enc_layers[0:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.decoder = dec_net

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self,x):
        """
        This method passes an image through all four encoder stages and
        returns the final feature map right after the relu4_1 layer
        """
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        return x

    def forward(self,content,style,alpha=1.0):
        """
        This method takes a batch of content images and a batch of
        style images of the same size and returns the stylized images
        The alpha value controls how strong the style is applied where
        one means full strength
        """
        with torch.no_grad():
            content_feat = self.encode(content)
            style_feat = self.encode(style)
            t = adaptive_instance_normalization(content_feat,style_feat)
            t = alpha*t+(1-alpha)*content_feat
            out = self.decoder(t)
        return out.clamp(0,1)
