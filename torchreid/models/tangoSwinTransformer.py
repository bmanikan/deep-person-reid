"""
Code source: TangoEYE - Nizam SwinTransformer
"""
from __future__ import division, absolute_import, print_function
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import init
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import fuse_conv_bn_eval
import pickle
import timm




__all__ = [
    'TangoSwinTransformer21'
]

#TODO
model_urls = {
    'TangoSwinTransformer21':
    '/home/trewq/Desktop/Projects/RetailAnalytics/REID/RESEARCH/REID_OFFICE/Model Experiments/PROD_Model/swin_2000C_m0.4_net_last.pth',
}


def fuse_all_conv_bn(model):
    stack = []
    for name, module in model.named_children():
        if list(module.named_children()):
            fuse_all_conv_bn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            if not stack:
                continue
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))
    return model


def load_state_dict_mute(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.
 
        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
 
        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []
 
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]
 
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
 
        load(self)
        del load
 
        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
 
 
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
 
 
def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True



class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=1024, loss='softmax'):
        super(ClassBlock, self).__init__()
        self.loss = loss
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
 
        classifier = []
        classifier += [nn.Linear(linear, class_num, bias=True)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
 
        self.add_block = add_block
        self.classifier = classifier
 
    def forward(self, x):
        x = self.add_block(x)

        if not self.training:
            return x

        if self.loss == 'triplet':
            f = x
            x = self.classifier(x)
            return x, f
        elif self.loss == 'softmax':
            x = self.classifier(x)
            return x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
            


class ft_net_swinv2(nn.Module):
 
    def __init__(self, class_num, input_size=(256, 128), droprate=0.5, stride=2, loss='softmax', linear_num=512):
        super(ft_net_swinv2, self).__init__()
        model_ft = timm.create_model('swinv2_base_window8_256', pretrained=False, img_size = input_size, drop_path_rate = 0.2)
        model_full = timm.create_model('swinv2_base_window8_256', pretrained=True)
        load_state_dict_mute(model_ft, model_full.state_dict(), strict=False)
        #model_ft = timm.create_model('swinv2_cr_small_224', pretrained=True, img_size = input_size, drop_path_rate = 0.2)
        # avg pooling to global pooling
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.loss = loss
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, loss = self.loss)
        print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x, return_featuremaps=False):
        x = self.model.forward_features(x)
        if return_featuremaps:
            return x
        x = self.avgpool(x.permute((0,2,1))) # B * 1024 * WinNum
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


def init_pretrained_weights(network, model_path):
    network.load_state_dict(torch.load(model_path))
    return network
    
 
 
def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def TangoSwinTransformer21(num_classes=2000, loss='softmax', pretrained=True, **kwargs):
    model = ft_net_swinv2(
    					  num_classes, 
    					  loss = loss,
    					  linear_num=512
    					  )

    if pretrained:
        model = init_pretrained_weights(model, model_urls['TangoSwinTransformer21'])
    return model