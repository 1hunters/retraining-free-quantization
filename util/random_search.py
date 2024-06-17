import torch
import numpy as np
from .greedy_search import forward_loss
from .utils import model_profiling, reset_batchnorm_stats
from .mpq import sample_one_mixed_policy

def do_random_search(loader, model, criterion, metrics, quan_scheduler, topK=30):
    acc_set = []
    conf_set = []
    model.eval()
    reset_batchnorm_stats(model)
    
    loader.sampler.set_epoch(0)
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.cuda(), targets.cuda()
    model.train()

    while len(conf_set) < 1000:
        conf = sample_one_mixed_policy(model, quan_scheduler)
        bitops, model_size = model_profiling(model)

        if metrics - .1 <= bitops <= metrics + .1:
            conf_set.append(conf)
            with torch.no_grad():
                _, top1_error = forward_loss(model, criterion, inputs, targets, None, return_acc=True)
            acc_set.append(1 - top1_error)
            print(len(conf_set), acc_set[-1])
    
    topk_conf = [conf_set[idx] for idx in np.array(acc_set).argsort()[-topK:]]

    eval_set = []
    for i, conf in enumerate(topk_conf):
        eval_set.append(
            [i, [pair[0] for pair in conf], [pair[1] for pair in conf]]
        )
    return eval_set
    


