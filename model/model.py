import logging
from .mobilenetv2 import mobilenet_v2
import timm
import torch
import torchvision
from torchvision.models import resnet101, resnet18

def create_model(arch, dataset='imagenet', pre_trained=True):
    logger = logging.getLogger()

    model = None
    if dataset == 'imagenet':
        if arch == 'resnet18':
            model = timm.create_model('gluon_resnet18_v1b', pretrained=True)
        elif arch == 'mobilenetv2':
            model = mobilenet_v2(pretrained=True)
        elif arch == 'resnet101':
            model = resnet101(torchvision.models.ResNet101_Weights)
        elif arch == 'efficientnet_lite':
            model = timm.create_model('efficientnet_lite0', pretrained=True)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (arch, dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (arch, dataset)
    msg += '\n          Use pre-trained model = %s' % pre_trained
    logger.info(msg)

    return model
