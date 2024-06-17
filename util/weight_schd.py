from typing import Any
import numpy as np

class CosineSched:
    def __init__(self, start_step, max_step, eta_start, eta_end, **xargs):
        assert start_step < max_step

        self.start_step = start_step
        self.max_step = max_step
        self.eta_start = eta_start
        self.eta_end = eta_end
    
    def _step(self, cur_step):
        pass

    def __call__(self, cur_step):
        if cur_step < self.start_step:
            return self.eta_start
        
        eta_cur_step = self.eta_end + 1/2 * (self.eta_start - self.eta_end) * (np.cos(np.pi * \
                                                                                    (cur_step - self.start_step)/(self.max_step - self.start_step) \
                                                                                    ) + 1)
    
        return eta_cur_step


class CosineTempDecay:
    def __init__(self, t_max, temp_range=(20.0, 2.0), rel_decay_start=0):
        self.t_max = t_max
        self.start_temp, self.end_temp = temp_range
        self.decay_start = rel_decay_start

    def __call__(self, t):
        if t < self.decay_start:
            return self.start_temp

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)
        return self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + np.cos(rel_t * np.pi))

class ConstantDecay:
    def __init__(self, t_max, temp_max, *arg, **kwds) -> None:
        self.t_max = t_max
        self.temp_max = temp_max
    
    def __call__(self, t):
        return self.temp_max