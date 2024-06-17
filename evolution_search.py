import random

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

import argparse
import os
import yaml

from util.qat import set_bit_width, model_profiling
from util.mpq import sample_one_mixed_policy
from process import validate

class EvolutionSearcher(object):

    def __init__(self, args, device, train_loader, model, val_loader, test_loader, output_dir, quantized_layers):
        self.quantized_layers = quantized_layers
        self.device = device
        self.model = model
        self.model_without_ddp = model.module
        self.args = args
        self.max_epochs = 20
        self.select_num = 10
        self.population_num = 50
        self.m_prob = 0.2
        self.crossover_num = 25
        self.mutation_num = 25
        self.train_loader = train_loader
        self.bops_limits = args.bops_limits
        self.min_bops_limits = args.min_bops_limits
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.s_prob = 0.4
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []

        print(random.choice(self.quantized_layers[0].weight_bit_cands).cpu().item())

        def _reset():
            for l in self.quantized_layers:
                l.reset_bits_cands()
                w_cand_min = min(l.weight_bit_cands)
                if w_cand_min == 2:
                    
                    print('remove 2bits')
                    l.set_bit_cands([6,5,4,3], bit_cands_a=[6,5,4,3])
        
        _reset()


    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        w_cand, a_cand = cand
        set_bit_width(self.model, w_cand, a_cand)
        bops, _ = model_profiling(self.model_without_ddp) # only support bops now
        info['bops'] =  bops

        if info['bops'] > self.bops_limits:
            # print(f'bops {bops} limit exceed')
            return False

        if info['bops'] < self.min_bops_limits:
            # print('under minimum bops limit')
            return False

        criterion = torch.nn.CrossEntropyLoss()

        # print('arch', cand, 'bops', bops)

        print("rank:", torch.distributed.get_rank(), cand, info['bops'])
        eval_acc = validate(self.test_loader, self.model, criterion, self.epoch, None,
                           self.args, train_loader=self.train_loader, eval_predefined_arch=[cand], train_mode=False)
        # eval_acc = [0]
        
        # test_acc = validate(self.test_loader, self.model, criterion, self.epoch, None,
        #                    self.args, train_loader=self.train_loader, eval_predefined_arch=[cand], train_mode=False)
        # info['test_acc'] = test_acc[0]
        
        # test_stats = evaluate(self.test_loader, self.model, self.device, amp=self.args.amp, mode='retrain', retrain_config=sampled_config)

        info['acc'] = eval_acc[0]
        # info['test_acc'] = test_acc[0]
        info['test_acc'] = 0

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):

        cand = sample_one_mixed_policy(self.model, self.args)
        cand = (tuple(cand[0]), tuple(cand[1]))
        return cand

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            # depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)

            w_cand, a_cand = list(cand[0]), list(cand[1])

            for i in range(len(w_cand)):
                random_s = random.random()
                if random_s < m_prob:
                    w_cand[i] = random.choice(self.quantized_layers[i].weight_bit_cands).cpu().item()
            
            for i in range(len(a_cand)):
                random_s = random.random()
                if random_s < m_prob:
                    a_cand[i] = random.choice(self.quantized_layers[i].act_bit_cands).cpu().item()

            return (tuple(w_cand), tuple(a_cand))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            while p1 == p2:
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            # max_iters_tmp = 50
            # while len(p1) != len(p2) and max_iters_tmp > 0:
            #     max_iters_tmp -= 1
            #     p1 = random.choice(self.keep_top_k[k])
            #     p2 = random.choice(self.keep_top_k[k])

            return (tuple(random.choice([i, j]) for i, j in zip(p1[0], p2[0])), tuple(random.choice([i, j]) for i, j in zip(p1[1], p2[1])))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.val_loader.sampler.set_epoch(self.epoch)
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, Top-1 test acc = {}, bops = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['test_acc'],self.vis_dict[cand]['bops']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()