import logging
import os

import torch
import torch.nn as nn
from quan import QuanConv2d
from timm.utils import unwrap_model, get_state_dict
from util.dist import master_only

logger = logging.getLogger()

@master_only
def save_checkpoint(epoch, arch, model, target_model, optimizer, extras=None, is_best=None, name=None, output_dir='.', lr_scheduler=None, lr_scheduler_q=None, optimizer_q=None):
    """Save a pyTorch training checkpoint
    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pyTorch model
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        output_dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(output_dir):
        raise IOError('Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    filepath = os.path.join(output_dir, filename)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    filepath_best = os.path.join(output_dir, filename_best)

    checkpoint = {
        'epoch': epoch,
        'state_dict': get_state_dict(model), 
        'arch': arch,
        'extras': extras,
        'optimizer': optimizer.state_dict(), 
        'state_dict_ema': get_state_dict(unwrap_model(target_model.ema)), 
        'lr_scheduler': lr_scheduler.state_dict(), 
        'lr_scheduler_q': lr_scheduler_q.state_dict(), 
        'optimizer_q': optimizer_q.state_dict(), 
    }

    msg = '([%d] Epoch) Saving checkpoint to:\n' % epoch
    msg += '             Current: %s\n' % filepath
    torch.save(checkpoint, filepath)
    if is_best:
        msg += '                Best: %s\n' % filepath_best
        torch.save(checkpoint, filepath_best)
    logger.info(msg)


def load_checkpoint(model:nn.Module, chkp_file, model_device=None, strict=True, lean=False, optimizer=None, override_optim=False, lr_scheduler=None, lr_scheduler_q=None, optimizer_q=None):
    """Load a pyTorch training checkpoint.
    Args:
        model: the pyTorch model to which we will load the parameters.  You can
        specify model=None if the checkpoint contains enough metadata to infer
        the model.  The order of the arguments is misleading and clunky, and is
        kept this way for backward compatibility.
        chkp_file: the checkpoint file
        lean: if set, read into model only 'state_dict' field
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, optimizer, start_epoch
    """
    if not os.path.isfile(chkp_file):
        raise IOError('Cannot find a checkpoint at', chkp_file)

    checkpoint = torch.load(chkp_file, map_location=lambda storage, loc: storage)

    if 'state_dict' not in checkpoint:
        raise ValueError('Checkpoint must contain model parameters')

    extras = checkpoint.get('extras', None)
    arch = checkpoint.get('arch', '_nameless_')

    # optimizer_state = checkpoint.get('optimizer', None)
    # if optimizer is not None and optimizer_state is not None and not override_optim:
    #     optimizer.load_state_dict(optimizer_state)

    optimizer_q_state = checkpoint.get('optimizer_q', None)
    if optimizer_q is not None and optimizer_q_state is not None and not override_optim:
        optimizer_q.load_state_dict(optimizer_q_state)
    
    lr_scheduler_state = checkpoint.get('lr_scheduler', None)
    if lr_scheduler is not None and lr_scheduler_state is not None:
        lr_scheduler.load_state_dict(lr_scheduler_state)

    lr_scheduler_q_state = checkpoint.get('lr_scheduler_q', None)
    if lr_scheduler_q is not None and lr_scheduler_q_state is not None:
        lr_scheduler_q.load_state_dict(lr_scheduler_q_state)
    
    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

    for name, module in model.module.named_modules():
        if isinstance(module, QuanConv2d) and hasattr(module, 'current_bit_cands') and name + '.' + 'current_bit_cands' in checkpoint['state_dict']:
            module.current_bit_cands = torch.ones(len(checkpoint['state_dict'][name + '.' + 'current_bit_cands']), device=module.weight.device, dtype=torch.int32)
        
        if isinstance(module, QuanConv2d) and hasattr(module, 'current_bit_cands_w') and name + '.' + 'current_bit_cands_w' in checkpoint['state_dict']:
            module.current_bit_cands_w = torch.ones(len(checkpoint['state_dict'][name + '.' + 'current_bit_cands_w']), device=module.weight.device, dtype=torch.int32) 
        
        if isinstance(module, QuanConv2d) and hasattr(module, 'current_bit_cands_a') and name + '.' + 'current_bit_cands_a' in checkpoint['state_dict']:
            module.current_bit_cands_a = torch.ones(len(checkpoint['state_dict'][name + '.' + 'current_bit_cands_a']), device=module.weight.device, dtype=torch.int32) 
            
        # if isinstance(module, QuanConv2d) 

    strict = False
    anomalous_keys = model.module.load_state_dict(checkpoint['state_dict'], strict)

    if strict:
        if anomalous_keys:
            missing_keys, unexpected_keys = anomalous_keys
            if unexpected_keys:
                logger.warning("The loaded checkpoint (%s) contains %d unexpected state keys" %
                            (chkp_file, len(unexpected_keys)))
            if missing_keys:
                print(missing_keys)
                raise ValueError("The loaded checkpoint (%s) is missing %d state keys" %
                                (chkp_file, len(missing_keys)))
            

    model.cuda()

    if lean:
        logger.info("Loaded checkpoint %s model (next epoch %d) from %s", arch, 0, chkp_file)
        return model, 0, None
    else:
        logger.info("Loaded checkpoint %s model (next epoch %d) from %s", arch, start_epoch, chkp_file)
        return model, start_epoch, extras
