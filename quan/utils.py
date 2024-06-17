import logging

from .func import *
from .quantizer import *
import torch
from torchvision.ops.misc import Conv2dNormActivation
from timm.models._efficientnet_blocks import InvertedResidual as InvertedResidual_EffNet
from model import InvertedResidual as InvertedResidual_MBb2

def quantizer(default_cfg, this_cfg=None, skip_quantization=False, remap_act=False, scale_grad=False):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v
    
    assert target_cfg['mode'] == 'lsq'

    q = IdentityQuan if skip_quantization else LsqQuan
    
    if remap_act:
        assert target_cfg['all_positive']
        target_cfg['all_positive'] = False

    target_cfg['scale_grad'] = scale_grad
    target_cfg.pop('mode')
    return q(**target_cfg)


def find_modules_to_quantize(model, configs):
    replaced_modules = dict()
    conv = None
    remap_act = False
    
    mb_blocks = (Conv2dNormActivation, InvertedResidual_MBb2, InvertedResidual_EffNet)

    for name, module in model.named_modules():
        if isinstance(module, mb_blocks): #handle MobileNetv2 Block
            idx = 0
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Conv2d) and (submodule.groups == 1) and submodule.in_channels > 3:
                    # if idx == 0:
                    remap_act = True 
                    print(name, subname)

        if 'conv_head' in name or 'features.18' in name: #handle last quantized layer in efficientnet
            remap_act = True

        if type(module) in ops.keys():
            if name in configs.quan.excepts:
                if isinstance(module, torch.nn.BatchNorm2d):
                    replaced_modules[name] = SwithableBatchNorm(module, module.num_features, None)
                else:
                    # print(name)
                    replaced_modules[name] = ops[type(module)](
                        module,
                        bits_list=[8],
                        quan_w_fn=quantizer(configs.quan.weight, scale_grad=getattr(configs, "scale_gradient", True)),
                        quan_a_fn=quantizer(configs.quan.act, skip_quantization=False, scale_grad=getattr(configs, "scale_gradient", True)),
                        fixed_bits=(8, 8) if isinstance(module, torch.nn.Linear) else (8, 32)
                    )
            else:
                if isinstance(module, torch.nn.BatchNorm2d):
                    replaced_modules[name] = SwithableBatchNorm(module, module.num_features, target_bits, conv)
                else:
                    target_bits = configs.target_bits
                    target_bits = configs.target_bits[:-1] if (min(configs.target_bits) == 2 and module.weight.shape[1] == 1) else configs.target_bits
                    
                    replaced_modules[name] = ops[type(module)](
                        module, 
                        bits_list=target_bits, 
                        quan_w_fn=quantizer(configs.quan.weight, scale_grad=getattr(configs, "scale_gradient", True)),
                        quan_a_fn=quantizer(configs.quan.act, remap_act=remap_act if module.weight.shape[1] != 1 else False, scale_grad=getattr(configs, "scale_gradient", True)),
                        fixed_bits = None,
                        split_aw_cands=configs.split_aw_cands
                    )
                    if remap_act:
                        remap_act = False
                    conv = replaced_modules[name]
        elif name in configs.quan.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: torch.nn.Module):
        for n, c in child.named_children():
            if type(c) in ops.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
