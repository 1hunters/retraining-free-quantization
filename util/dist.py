# from logging import Logger, _ExcInfoType, _Level, WARNING
# import sys, os, time, io, traceback, warnings, weakref, collections.abc
# from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
import torch.distributed as distributed
import os 
import torch

def init_dist_nccl_backend(configs):
    configs.distributed = 'WORLD_SIZE' in os.environ
    
    configs.device = 'cuda:0'
    configs.world_size = 1
    configs.rank = 0

    if configs.distributed:
        configs.device = 'cuda:%d' % configs.local_rank
        torch.cuda.set_device(configs.local_rank)
        distributed.init_process_group(backend='nccl', init_method='env://')
        configs.world_size = distributed.get_world_size()
        configs.rank = distributed.get_rank()

def setup_print(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_master():
    return distributed.is_initialized() and distributed.get_rank() == 0

def master_only(fn):
    def wrapper(*args,**kwargs):
        if is_master():
            fn(*args,**kwargs)
        else:
            return None
    
    return wrapper 

@master_only
def logger_info(logger, msg, *args):
    if logger is not None:
        logger.info(msg, *args)

@master_only
def tbmonitor_add_scalars(tbm, main_tag, scalar, epoch):
    tbm.writer.add_scalars(main_tag, scalar, epoch)