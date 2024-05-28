import torch
from torch import nn
import torch.nn.functional as F


# GeM pooling layer
class GeM(nn.Module):
    """Generalized Mean Pooling (GeM) pooling layer.
    GeM pooling is intermediate between avgpool and maxpool. i.e., max pooling when pk→∞ and average pooling for pk = 1.
    this pk is hyperparameter and differentiable. this parameter can be learnt using backprop.
    below code, p is initialized as nn.Parameter and hence it is learned during backprop to deviate from initial value 3.

    Reference:
        Implementation: https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter?scriptVersionId=88427415&cellId=27
        Explanation: https://amaarora.github.io/posts/2020-08-30-gempool.html#gem-pooling 

    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()       
        self.p = nn.Parameter(torch.ones(1) * p)  
        self.eps = eps  # constant to avoid division by zero

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)  
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)  
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
