"""
This file holds the gradient reversal layer and the small domain
discriminator network shared by dann and cdan
"""

import math
import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    """
    This autograd function copies its input on the forward pass and
    flips the sign of the gradient on the backward pass after scaling it
    by alpha
    """

    @staticmethod
    def forward(ctx,x,alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        return -ctx.alpha*grad_output,None

class GradientReversal(nn.Module):
    """
    This module applies the gradient reversal function with a given
    alpha so it can sit inside a normal forward pass
    """

    def forward(self,x,alpha):
        """
        This method passes x through unchanged and stores alpha for the
        backward pass to use
        """
        return GradientReversalFunction.apply(x,alpha)

class DomainDiscriminator(nn.Module):
    """
    This class is the small binary domain classifier used by dann and
    cdan. It sits behind the gradient reversal layer and tries to tell
    source features from target features
    """

    def __init__(self,input_dim,hidden_dim=256,dropout=0.5):
        super().__init__()
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,2),
        )

    def forward(self,x,alpha):
        """
        This method reverses the gradient of x using the given alpha and
        then runs the result through the domain classifier
        """
        x = self.grl(x,alpha)
        return self.net(x)

def grl_alpha(progress,max_alpha=1.0):
    """
    This function computes the gradient reversal strength for the given
    training progress between zero and one following the standard dann
    schedule scaled by the chosen maximum strength
    """
    return max_alpha*(2.0/(1.0+math.exp(-10.0*progress))-1.0)
