import torch 
import torch.distributed as dist
from .quantizer import Quantizer
import random

# code from https://github.com/zhutmost/lsq-net

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y-y_grad).detach()+y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y-y_grad).detach()+y_grad


def quantize(x, s, thd_neg, thd_pos, scale_grad=False):
    if scale_grad:
        s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(s, s_grad_scale)
    else:
        s_scale = s
    
    x = x / s_scale
    x = x.clamp(min=thd_neg, max=thd_pos)
    x = round_pass(x) 
    x = x * s_scale
    return x


def compute_thd(self, bits):
    if self.all_positive:
        assert not self.symmetric, "Positive quantization cannot be symmetric"
        # unsigned activation is quantized to [0, 2^b-1]
        thd_neg = 0
        thd_pos = 2 ** bits - 1
    else:
        if self.symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            thd_neg = - 2 ** (bits - 1) + 1
            thd_pos = 2 ** (bits - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            thd_neg = - 2 ** (bits - 1)
            thd_pos = 2 ** (bits - 1) - 1
    
    if isinstance(thd_neg, torch.Tensor):
        thd_neg = int(thd_neg.cpu().item())
        thd_pos = int(thd_pos.cpu().item())
    elif isinstance(thd_neg, float):
        thd_neg = int(thd_neg)
        thd_pos = int(thd_pos)
    
    return thd_neg, thd_pos


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, scale_grad=True):
        super().__init__(bit)
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1))
        self.bit_list = (8, )
        self.skip_init = False
        self.bit_mapping = {
            8: 0
        }
        self.bits = 2
        self.scale_gradient = scale_grad
        self.using_one_scale = True
    
    def init_from(self, x, bit_list, *args, **kwargs):
        self.bit_list = tuple(bit_list)
        mapping = {

        }

        for bit_idx, bit in enumerate(self.bit_list):
            mapping[bit] = bit_idx
        
        self.bit_mapping = mapping
        self.register_buffer('init_state', torch.zeros(len(bit_list)))

        if x is not None:
            if self.per_channel:
                self.s = torch.nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            else:
                self.s = torch.nn.Parameter(torch.ones(len(bit_list)))
                mean = x.detach().mean() 
                std = x.detach().std() 
                for i, b in enumerate(self.bit_list): 
                    s_init = torch.max((mean-3*std).abs(), (mean+3*std).abs())/2**(max(bit_list)-1) * 2 ** (max(bit_list) - b)

                    self.s[i].data.copy_(s_init)
                    self.init_state[i].fill_(1)
        else:
            self.s = torch.nn.Parameter(torch.ones(len(bit_list)))

    def __repr__(self):
        return f'LSQ quantizer. Bit-width candidates: {self.bit_list}, all positive: {self.all_positive}, symmetric: {self.symmetric}, gradient scaling: {self.scale_gradient}'

    def _index_bits(self, bits):
        if isinstance(bits, torch.Tensor):
            bits = bits.cpu().item()
        return self.bit_mapping[bits]
    
    def sample(self, cands, max=False, min=False):
        if max:
            bit_width = cands.max()
        elif min:
            bit_width = cands.min()
        else:
            bit_width = random.choice(cands)

        return bit_width.cpu().item()
        
    def get_scale(self, bit_width, detach=True):
        s = self.s[self._index_bits(bit_width)]
        if detach:
            s = s.detach()

        return s
    
    def get_max_bit_width_and_scale(self, detach=True):
        max_bit_witdh = max(self.bit_list)
        return max_bit_witdh, self.get_scale(bit_width=max(self.bit_list), detach=detach)
    
    def weight_bound(self, bits, scale=None):
        lower, upper = compute_thd(self, bits)
        if scale is None:
            step_size = self.get_scale(bits)
        else:
            step_size = scale
        return step_size * lower, step_size * upper

    def forward(self, x, bits, is_activation=False, skip_init=False, scale=None, **args):
        if bits is None or bits >= 32:
            self.bit_mapping = {
                32: 0
            }
            return x
        
        idx = self._index_bits(bits=bits)
        thd_neg, thd_pos = compute_thd(self, bits)
        self.bits = bits

        if self.init_state[idx] == 0 and not skip_init:
            self.init_state[idx].fill_(1)
            s_init = x.detach().abs().mean() * 2 / (thd_pos ** 0.5)

            if dist.get_world_size() > 1:
                dist.all_reduce(s_init)
                s_init /= dist.get_world_size()
            
            self.s[idx].data.copy_(s_init)
        if scale is None:
            s = self.s[idx]
        else:
            s = scale
            
        x = quantize(x, s, thd_neg, thd_pos, scale_grad=self.scale_gradient)
        return x