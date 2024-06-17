import torch.nn as nn


class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()
        self.x_min_fp32 = self.x_max_fp32 = None

    def init_from(self, x, bit_list, *args, **kwargs):
        pass

    def forward(self, x, bits, is_activation=False, floor_tensor=False):
        raise NotImplementedError

    def set_quant_range(self, x_min, x_max):
        pass


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        # assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x, bits, is_activation=False, floor_tensor=False):
        return x
    
    def init_from(self, x, bit_list, *args, **kwargs):
        return
