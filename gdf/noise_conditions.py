import torch
import numpy as np

class BaseNoiseCond():
    def __init__(self, *args, shift=1, clamp_range=[-1e9, 1e9], **kwargs):
        self.shift = shift
        self.clamp_range = clamp_range
        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass # this method is optional, override it if required
    
    def cond(self, logSNR):
        raise Exception("this method needs to be overriden")

    def __call__(self, logSNR):
        if self.shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(self.shift)
        return self.cond(logSNR).clamp(*self.clamp_range)

class CosineTNoiseCond(BaseNoiseCond):
    def setup(self, s=0.008, clamp_range=[0, 1]): # [0.0001, 0.9999]
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2
        
    def cond(self, logSNR):
        var = logSNR.sigmoid()
        var = var.clamp(*self.clamp_range)
        s, min_var = self.s.to(var.device), self.min_var.to(var.device)
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t

class EDMNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return -logSNR/8
    
class SigmoidNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return (-logSNR).sigmoid()

class LogSNRNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return logSNR