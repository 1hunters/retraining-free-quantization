from copy import deepcopy
import torch
from quan.func import QuanConv2d,QuanLinear,SwithableBatchNorm
from timm.utils import reduce_tensor, distribute_bn
from torch.distributed import get_world_size
import numpy as np
import random
import torch.nn as nn
from timm.scheduler import create_scheduler

def create_optimizer_and_lr_scheduler(model, configs):
    all_parameters = model.parameters()
    weight_parameters = []
    bn_parameters = []
    quant_parameters = []

    for pname, p in model.named_parameters():
        if p.ndimension() == 4 and 'bias' not in pname:
            # print('weight_param:', pname)
            weight_parameters.append(p)
        if 'quan_a_fn.s' in pname or 'quan_w_fn.s' in pname or 'quan3.a' in pname or 'scale' in pname or 'start' in pname:
            # print('alpha_param:', pname)
            quant_parameters.append(p)

    weight_parameters_id = list(map(id, weight_parameters))
    alpha_parameters_id = list(map(id, quant_parameters))
    other_parameters1 = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1))

    quantizer_optim = 'adam'
    if quantizer_optim == 'adam':
        optimizer_q = torch.optim.Adam(
            [
                {'params' : quant_parameters, 'lr': getattr(configs, 'q_lr', 1e-5)}
            ],
            lr = getattr(configs, 'q_lr', 1e-5)
        )

        optimizer = torch.optim.SGD(
            [
                {'params' : weight_parameters, 'weight_decay': configs.weight_decay, 'lr': configs.lr},
                {'params' : other_parameters, 'lr': configs.lr},
            ],
            nesterov=True,
            momentum=configs.momentum
        )

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.epochs, eta_min=0)
        lr_scheduler, _ = create_scheduler(configs, optimizer)
        lr_scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=configs.epochs, eta_min=0)
    else:
        optimizer = torch.optim.Adam(
            [   {'params' : weight_parameters, 'weight_decay': configs.weight_decay, 'lr': configs.lr},
                {'params' : quant_parameters, 'lr': configs.lr*.1},
                {'params' : other_parameters, 'lr': configs.lr},
                
            ],
                betas=(0.9, 0.999)
        )
        
        optimizer_q, lr_scheduler_q = None, None
    
    return optimizer, optimizer_q, lr_scheduler, lr_scheduler_q

def preprocess_model(model, configs):
    dropout_p = getattr(configs, 'dropout', .0)

    if dropout_p > 0.:
        return model
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            
                print('droupout -> ', module.p)
        
        if 'mobilenet' in configs.arch:
            model.classifier = model.classifier[1:]
    
    return model

def model_profiling(model: torch.nn.Module, first_last_layer_act_bits=8, first_last_layer_weight_bits=8, return_layers=False):
    bitops = 0.
    model_size = 0.
    quantized_layers = []
    bn = []
    next_bn = False

    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d):
            if hasattr(module, 'bits') and (module.bits is not None and len(module.bits) > 1):
                # module: torch.nn.Conv2d
                assert isinstance(module.bits, (list, tuple))
                next_bn = True
                quantized_layers.append(module)

                wbits, abits = module.bits
                bitops += (wbits*abits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels*module.output_size)//module.groups
                model_size += (wbits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels)//module.groups
            
            elif module.bits is None:
                bitops += first_last_layer_act_bits*first_last_layer_weight_bits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels*module.output_size
                
                model_size += first_last_layer_weight_bits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels
        
        if isinstance(module, QuanLinear):
            bitops += first_last_layer_act_bits*first_last_layer_weight_bits*module.in_features*module.out_features
            model_size += first_last_layer_weight_bits*module.in_features*module.out_features
        
        if isinstance(module, SwithableBatchNorm) and next_bn:
            bn.append(module)
            next_bn = False
    
    bitops /= 1e9
    model_size /= (8*1024*1024)
    
    if return_layers:
        assert len(quantized_layers) == len(bn)
        return bitops, model_size, quantized_layers, bn
    else:
        return bitops, model_size

def reset_batchnorm_stats(m):
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        m.momentum = None

@torch.no_grad()
def calibrate_batchnorm_state(model, loader, num_batch=30, reset=False, distributed_training=True, epoch=0):

    if epoch >= 0 and hasattr(loader, 'sampler'):
        loader.sampler.set_epoch(epoch)
    model.eval()

    if reset:
        for _, module in model.named_modules():
            reset_batchnorm_stats(module)

    for batch_idx, (inputs, _) in enumerate(loader):
            if batch_idx > num_batch:
                break
            
            # print(batch_idx)
            
            inputs = inputs.cuda()
            model(inputs)
        
    if distributed_training: # all reduce for each GPU
        distribute_bn(model, world_size=get_world_size(), reduce=True)
    
    model.eval()

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_meter(meter, loss, QE_loss, dist_loss, IDM_loss, acc1, acc5, size, batch_time, world_size):
    data = torch.cat([loss.data.reshape(1), acc1.reshape(1), acc5.reshape(1), 
                QE_loss.data.reshape(1) if QE_loss is not None and QE_loss != 0 else torch.zeros(1, device=loss.device), 
                dist_loss.data.reshape(1) if dist_loss is not None and dist_loss != 0 else torch.zeros(1, device=loss.device), 
                IDM_loss.data.reshape(1) if IDM_loss is not None and IDM_loss != 0 else torch.zeros(1, device=loss.device), 
                 ])
    reduced_data = reduce_tensor(data, world_size)
    reduced_loss, reduced_top1, reduced_top5, reduced_QE_loss, reduced_dist_loss, reduced_IDM_loss = reduced_data
    
    meter['dist_loss'].update(reduced_dist_loss.item(), size)
    meter['IDM_loss'].update(reduced_IDM_loss.item(), size)
    meter['QE_loss'].update(reduced_QE_loss.item(), size)
    meter['loss'].update(reduced_loss.item(), size)
    meter['top1'].update(reduced_top1.item(), size)
    meter['top5'].update(reduced_top5.item(), size)
    meter['batch_time'].update(batch_time)