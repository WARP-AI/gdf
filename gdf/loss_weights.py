import torch
import numpy as np

# --- Loss Weighting
class BaseLossWeight():
    def weight(self, *args, **kwargs):
        raise Exception("this method needs to be overriden")
        
    def __call__(self, logSNR, *args, shift=1, **kwargs): 
        if shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(shift)
        return self.weight(logSNR, *args, **kwargs)

class ComposedLossWeight(BaseLossWeight):
    def __init__(self, div, mul):
        self.mul = [mul] if isinstance(mul, BaseLossWeight) else mul       
        self.div = [div] if isinstance(div, BaseLossWeight) else div       

    def weight(self, logSNR):
        prod, div = 1, 1
        for m in self.mul:
            prod *= m.weight(logSNR)
        for d in self.div:
            div *= d.weight(logSNR)
        return prod/div

class ConstantLossWeight(BaseLossWeight):
    def __init__(self, v=1):
        self.v = v
        
    def weight(self, logSNR):
        return torch.ones_like(logSNR) * self.v
    
class SNRLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp()
    
class P2LossWeight(BaseLossWeight):
    def weight(self, logSNR, k=1.0, gamma=1.0):
        return (k + logSNR.exp()) ** -gamma
    
class SNRPlusOneLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp() + 1
    
class MinSNRLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr
    
    def weight(self, logSNR):
        return logSNR.exp().clamp(max=self.max_snr)

class MinSNRPlusOneLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr
    
    def weight(self, logSNR):
        return (logSNR.exp() + 1).clamp(max=self.max_snr)
    
class TruncatedSNRLossWeight(BaseLossWeight):
    def __init__(self, min_snr=1):
        self.min_snr = min_snr
    
    def weight(self, logSNR):
        return logSNR.exp().clamp(min=self.min_snr)

class SechLossWeight(BaseLossWeight):
    def __init__(self, div=2):
        self.div = div
    
    def weight(self, logSNR):
        return 1/(logSNR/self.div).cosh()

class DebiasedLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return 1/logSNR.exp().sqrt()