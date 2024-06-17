import torch
import copy
import torch.nn.functional as F
from timm.layers.norm_act import BatchNormAct2d

class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, bits_list=[8], quan_w_fn=None, quan_a_fn=None, fixed_bits=None, split_aw_cands=False):
        # assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode
                         )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.bits = None
        self.weight = torch.nn.Parameter(m.weight.detach())

        self.output_size = None
        self.fixed_bits = fixed_bits

        self.split_aw_cands = split_aw_cands

        if not split_aw_cands:
            print(f"layer {self._get_name()} not using splited a_w cands!")
            self.register_buffer('current_bit_cands', torch.tensor(bits_list, dtype=torch.int))
        else:
            self.register_buffer('current_bit_cands_w', torch.tensor(bits_list, dtype=torch.int))
            self.register_buffer('current_bit_cands_a', torch.tensor(bits_list, dtype=torch.int))
        self.bits_list = bits_list

        self.quan_w_fn.init_from(m.weight, bits_list if fixed_bits is None else (fixed_bits[0], ))
        self.quan_a_fn.init_from(None, bits_list if fixed_bits is None else (fixed_bits[1], ))

        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def set_bit_cands(self, bit_cands, bit_cands_a=None):
        if isinstance(bit_cands, (list, tuple)):
            bit_cands = torch.tensor(bit_cands).to(self.weight.device, dtype=torch.int)
            if bit_cands_a is not None:
                bit_cands_a = torch.tensor(bit_cands_a).to(self.weight.device, dtype=torch.int)
        
        if not self.split_aw_cands:
            self.current_bit_cands = bit_cands
        else:
            self.current_bit_cands_w = bit_cands
            if bit_cands_a is not None:
                self.current_bit_cands_a = bit_cands_a
    
    def set_sampled_bit(self, bit_pair):
        self.bits = bit_pair
    
    def reset_bits_cands(self):
        self.set_bit_cands(self.bits_list, self.bits_list)
        return self.weight_bit_cands
    
    @property
    def weight_bit_cands(self):
        if self.split_aw_cands:
            return self.current_bit_cands_w
        else:
            return self.current_bit_cands
    
    @property
    def act_bit_cands(self):
        if self.split_aw_cands:
            return self.current_bit_cands_a
        else:
            return self.current_bit_cands
    
    @property
    def is_sample_min(self):
        wbits = self.bits[0]
        return min(self.weight_bit_cands) == wbits
    
    def sample_bit_conf(self, act_fp=False, weight_fp=False, full_mixed=True, max_sample_bits=None, sample_max=False, sample_min=False):

        if act_fp:
            return (self.quan_w_fn.sample(self.weight_bit_cands), 32)
        
        if weight_fp:
            return (32, self.quan_a_fn.sample(self.act_bit_cands))
        
        if full_mixed:

            return (self.quan_w_fn.sample(self.weight_bit_cands, max=sample_max, min=sample_min), \
                     self.quan_a_fn.sample(self.act_bit_cands, max=sample_max, min=sample_min))
        else:
            raise NotImplementedError
    
    def __repr__(self):
        bit_cands_str = f"Dynamic Bit-width cands: using splitted cands {self.split_aw_cands}, weights cands {self.weight_bit_cands.cpu().tolist()}, act cands {self.act_bit_cands.cpu().tolist()}"
        return super().__repr__()[:-1] + bit_cands_str + "\n )"
    
    def forward(self, x):
        weight = self.weight

        if self.bits is not None or self.fixed_bits is not None:
            if self.fixed_bits is not None:
                wbits, abits = self.fixed_bits
            else:
                wbits, abits = self.bits

            weight = self.quan_w_fn(weight, wbits, is_activation=False)
            x = self.quan_a_fn(x, abits, is_activation=True)

        out = F.conv2d(x, weight=weight, stride=self.stride, padding=self.padding, groups=self.groups)

        self.output_size = out.shape[-1]**2
        return out

class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, bits_list=[], quan_w_fn=None, quan_a_fn=None, fixed_bits=None, split_aw_cands=False):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.bits = None
        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight, bits_list)
        self.quan_a_fn.init_from(None, bits_list)

        self.fixed_bits = fixed_bits
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        

    def forward(self, x):
        if self.bits is not None or self.fixed_bits is not None:
            if self.fixed_bits is not None:
                wbits, abits = self.fixed_bits
            else:
                wbits, abits = self.bits
            weight = self.quan_w_fn(self.weight, wbits, is_activation=False)
            x = self.quan_a_fn(x, abits, floor_tensor=False, is_activation=True)

            bias = self.bias
            
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)

class SwithableBatchNorm(torch.nn.Module):
    def __init__(self, m : torch.nn.BatchNorm2d, num_features, bits_list=None, conv=None):
        super(SwithableBatchNorm, self).__init__()
        self.num_features = num_features
        self.bits_list = bits_list
        self.bit_width = -1
        self._is_sample_min = False
        
        if bits_list is not None:
            bn_list = [torch.nn.ModuleList([copy.deepcopy(m) for _ in range(len(bits_list))]) \
            for _ in range(len(bits_list))]
            self.bn_list = torch.nn.ModuleList(bn_list)
        else:
            self.bn = copy.deepcopy(m)
                
        
        self.size = len(self.bn_list)*len(self.bn_list) if self.bits_list is not None else 1
    
    @property
    def is_sample_min(self):
        return self._is_sample_min
    
    def switch_bn(self, bit_width, is_sample_min=False):
        if self.bits_list is not None:
            self.bit_width = bit_width
            self._is_sample_min = is_sample_min

    def forward(self, x):
        if self.bits_list is not None:
            if not isinstance(self.bit_width, (list, tuple)):
                i = j = 0
            else:
                w_bits, a_bits = self.bit_width

                i = self.bits_list.index(w_bits)
                j = self.bits_list.index(a_bits)
            
            x = self.bn_list[i][j](x)

            return x
        
        return self.bn(x)

    def __repr__(self):
        if self.bits_list is not None:
            has_act_layer = hasattr(self.bn_list[0][0], 'act')
            if has_act_layer:
                act_layer = type(getattr(self.bn_list[0][0], 'act'))
            else:
                act_layer = 'None'
        else:
            has_act_layer = hasattr(self.bn, 'act')
            if has_act_layer:
                act_layer = type(getattr(self.bn, 'act'))
            else:
                act_layer = 'None'
        return f'SwithableBatchNorm: {self.size} Layers' + f', Act layer: {act_layer}'

ops = {
    torch.nn.Conv2d: QuanConv2d,
    torch.nn.Linear: QuanLinear,
    torch.nn.BatchNorm2d: SwithableBatchNorm, 
    BatchNormAct2d: SwithableBatchNorm,
}