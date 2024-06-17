from quan.func import SwithableBatchNorm
from timm.utils import reduce_tensor, unwrap_model
import torch
from quan.func import QuanConv2d

def switch_bit_width_bn(model, wbit, abits):
    for _, module in model.named_modules():
        if isinstance(module, (SwithableBatchNorm)):
            module.switch_bn((wbit, abits))

def switch_bit_width(model, quan_scheduler, wbit, abits):
    for name, module in unwrap_model(model).named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name not in quan_scheduler.excepts:
                module.bits = (wbit, abits)
    switch_bit_width_bn(model, wbit, abits)

def sample_max_cands(model, configs):
    sample_one_mixed_policy(model, configs, sample_max=True)

def sample_min_cands(model, configs):
    sample_one_mixed_policy(model, configs, sample_min=True)

def sample_one_mixed_policy(model, configs, max_sample_bits=None, sample_max=False, sample_min=False, weight_cands=None, act_cands=None):
    next_bn = False
    conf = []

    quan_scheduler = configs.quan

    weights, act, is_min = [], [], []
    quantized_layer_id = 0
    for idx, (name, module) in enumerate(unwrap_model(model).named_modules()):

        if isinstance(module, QuanConv2d):
            if name not in quan_scheduler.excepts:
                next_bn = True
                if weight_cands is not None and act_cands is not None:
                    bit_pair = (weight_cands[quantized_layer_id], act_cands[quantized_layer_id])
                else:
                    bit_pair = module.sample_bit_conf(act_fp=False, weight_fp=False, full_mixed=True, max_sample_bits=max_sample_bits, sample_max=sample_max, sample_min=sample_min)
                    
                module.set_sampled_bit(bit_pair)
                weights.append(bit_pair[0])
                act.append(bit_pair[1])
                # module.bits = bit_pair
                conf.append(bit_pair)
                is_sample_min = module.is_sample_min
                if is_sample_min:
                    is_min.append(quantized_layer_id)

                quantized_layer_id += 1

        elif isinstance(module, SwithableBatchNorm):
            if name not in quan_scheduler.excepts and next_bn:
                module.switch_bn(bit_pair, is_sample_min=is_sample_min)
                next_bn = False
    
    return weights, act, is_min