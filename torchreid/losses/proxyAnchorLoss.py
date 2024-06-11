import torch
import torch.nn as nn
import torch.nn.functional as F
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

def binarize(labels, num_classes):
    labels = labels.cpu().numpy()
    one_hot_labels = label_binarize(labels, classes=range(0, num_classes))
    one_hot_labels = torch.FloatTensor(one_hot_labels).cuda()
    return one_hot_labels

def l2_norm(tensor):
    tensor_size = tensor.size()
    squared_tensor = torch.pow(tensor, 2)
    sum_squared = torch.sum(squared_tensor, 1).add_(1e-12)
    norm = torch.sqrt(sum_squared)
    normalized_tensor = torch.div(tensor, norm.view(-1, 1).expand_as(tensor))
    return normalized_tensor.view(tensor_size)

class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, margin=0.1, alpha=32, use_gpu=True):
        super(ProxyAnchorLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha
        
        if use_gpu:
            self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim).cuda())
        else:
            self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))

        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    def forward(self, embeddings, labels):
        normalized_proxies = l2_norm(self.proxies)
        normalized_embeddings = l2_norm(embeddings)
        cosine_similarity = F.linear(normalized_embeddings, normalized_proxies)
        
        positive_one_hot = binarize(labels, num_classes=self.num_classes)
        negative_one_hot = 1 - positive_one_hot
    
        positive_exp = torch.exp(-self.alpha * (cosine_similarity - self.margin))
        negative_exp = torch.exp(self.alpha * (cosine_similarity + self.margin))

        positive_proxies = torch.nonzero(positive_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_positive_proxies = len(positive_proxies)
        
        positive_similarity_sum = torch.where(positive_one_hot == 1, positive_exp, torch.zeros_like(positive_exp)).sum(dim=0)
        negative_similarity_sum = torch.where(negative_one_hot == 1, negative_exp, torch.zeros_like(negative_exp)).sum(dim=0)
        
        positive_term = torch.log(1 + positive_similarity_sum).sum() / num_positive_proxies
        negative_term = torch.log(1 + negative_similarity_sum).sum() / self.num_classes
        
        loss = positive_term + negative_term
        return loss
