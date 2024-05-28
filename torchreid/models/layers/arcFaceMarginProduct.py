import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    r"""Implement large margin arc distance.
        Improves discriminative power of deep features by adding an angular margin to the decision boundary in the feature space.
        The goal is to increase the inter-class variance while reducing the intra-class variance, 
        thereby making the features more separable and improving classification performance.
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        reference: https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter?scriptVersionId=88427415&cellId=29
        

    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        
        # Initialize parameters
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.ls_eps = ls_eps  
        
        # Weight parameter (learnable)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # Xavier uniform initialization
        
        # Margin-related constants
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        # Normalize input and weights, then compute cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # Compute sine(theta)
        phi = cosine * self.cos_m - sine * self.sin_m  # Compute cos(theta + m)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)  # Easy margin
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # Hard margin

        # --------------------------- convert label to one-hot ---------------------
        # Create one-hot encoded labels
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply label smoothing if ls_eps > 0
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # --------------------- Combine phi and cosine with one-hot -----------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Scale output

        return output
