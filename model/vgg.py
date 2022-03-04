'''
    dummy file to use as an adaptor to switch between
    two vgg architectures

    vgg9: use vgg9_only.py which is from https://github.com/kuangliu/pytorch-cifar
    vgg11/13/16/19: use vgg_modified.py which is modified from https://github.com/pytorch/vision.git
'''

import torch
import torch.nn as nn
import model.vgg9_only as vgg9
import model.vgg_modified as vgg_mod
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_vgg_model(vgg_name, num_class):
    logging.info("GET_VGG_MODEL: Fetch {}".format(vgg_name))
    if vgg_name == 'vgg9':
        return vgg9.VGG('VGG9', num_class=num_class)
    elif vgg_name == 'vgg11':
        return vgg_mod.vgg11(num_class=num_class)
    elif vgg_name == 'vgg11_bn':
        return vgg_mod.vgg11_bn(num_class=num_class)
    elif vgg_name == 'vgg13':
        return vgg_mod.vgg13(num_class=num_class)
    elif vgg_name == 'vgg13_bn':
        return vgg_mod.vgg13_bn(num_class=num_class)
    elif vgg_name == 'vgg16':
        return vgg_mod.vgg16(num_class=num_class)
    elif vgg_name == 'vgg16_bn':
        return vgg_mod.vgg16_bn(num_class=num_class)
    elif vgg_name == 'vgg19':
        return vgg_mod.vgg19(num_class=num_class)
    elif vgg_name == 'vgg19_bn':
        return vgg_mod.vgg19_bn(num_class=num_class)
