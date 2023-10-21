import torch
import numpy as np

class CosineTNoiseCond():
    def __init__(self, s=0.008, clamp_range=[0, 1], shift=1): # [0.0001, 0.9999]
        self.s = torch.tensor([s])
        self.shift = shift
        self.clamp_range = clamp_range
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2
        
    def __call__(self, logSNR):
        if self.shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(self.shift)
        var = logSNR.sigmoid()
        var = var.clamp(*self.clamp_range)
        s, min_var = self.s.to(var.device), self.min_var.to(var.device)
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t

class EDMNoiseCond():
    def __call__(self, logSNR):
        return -logSNR/8
    
class SigmoidNoiseCond():
    def __call__(self, logSNR):
        return (-logSNR).sigmoid()

class LogSNRNoiseCond():
    def __call__(self, logSNR):
        return logSNR