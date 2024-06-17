import os

import numpy as np
import torch
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader


def init_dataloader(cfg, arch):
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    if cfg.dataset == 'imagenet':
        traindir = os.path.join(cfg.path, 'train')
        testdir = os.path.join(cfg.path, 'val')
        valdir = os.path.join(cfg.path, 'subImageNet') 

        if not os.path.exists(valdir): #using training set for searching?
            valdir = traindir

        train_set = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ]))

        test_set = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ]))
    else:
        raise NotImplementedError
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=(train_sampler is None), drop_last=True,
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.batch_size, shuffle=False, sampler=test_sampler, drop_last=False,
        num_workers=cfg.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=(val_sampler is None),drop_last=True,
        num_workers=cfg.workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler