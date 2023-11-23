import torch
import numpy as np

class BaseSchedule():
    def schedule(self, *args, **kwargs):
        raise Exception("this method needs to be overriden")
        
    def __call__(self, t, *args, shift=1, **kwargs):
        if isinstance(t, torch.Tensor):
            batch_size = None
        else:
            batch_size = t
            t = None
        logSNR = self.schedule(t.clamp(0, 1), batch_size, *args, **kwargs)
        if shift != 1:
            logSNR += 2 * np.log(1/shift)
        return logSNR

class CosineSchedule(BaseSchedule):
    def __init__(self, s=0.008, clamp_range=[0.0001, 0.9999]):
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def schedule(self, t, batch_size):
        if t is None:
            t = (1-torch.rand(batch_size)).add(0.001).clamp(0.001, 1.0)
        s, min_var = self.s.to(t.device), self.min_var.to(t.device)
        var = torch.cos((s + t)/(1+s) * torch.pi * 0.5).clamp(0, 1) ** 2 / min_var
        var = var.clamp(*self.clamp_range)
        logSNR = (var/(1-var)).log()
        return logSNR
    
class CosineSchedule2(BaseSchedule):
    def __init__(self, logsnr_range=[-15, 15]):
        self.t_min = np.arctan(np.exp(-0.5 * logsnr_range[1]))
        self.t_max = np.arctan(np.exp(-0.5 * logsnr_range[0]))

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        return -2 * (self.t_min + t*(self.t_max-self.t_min)).tan().log()
    
class SqrtSchedule(BaseSchedule):
    def __init__(self, s=1e-4, clamp_range=[0.0001, 0.9999]):
        self.s = s
        self.clamp_range = clamp_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        var = 1 - (t + self.s)**0.5
        var = var.clamp(*self.clamp_range)
        logSNR = (var/(1-var)).log()
        return logSNR

class RectifiedFlowsSchedule(BaseSchedule):
    def __init__(self, logsnr_range=[-15, 15]):
        self.logsnr_range = logsnr_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        logSNR = (((1-t)**2)/(t**2)).log().clamp(*self.logsnr_range)
        return logSNR

class EDMSampleSchedule(BaseSchedule):
    def __init__(self, sigma_range=[0.002, 80], p=7):
        self.sigma_range = sigma_range
        self.p = p

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        smin, smax, p = *self.sigma_range, self.p
        sigma = (smax ** (1/p) + (1-t) * (smin ** (1/p) - smax ** (1/p))) ** p
        logSNR = (1/sigma**2).log()
        return logSNR

class EDMTrainSchedule(BaseSchedule):
    def __init__(self, mu=-1.2, std=1.2):
        self.mu = mu
        self.std = std
        
    def schedule(self, t, batch_size):
        if t is not None:
            raise Exception("EDMTrainSchedule doesn't support passing timesteps: t")
        logSNR = -2*(torch.randn(batch_size) * self.std - self.mu)
        return logSNR

class LinearSchedule(BaseSchedule):
    def __init__(self, logsnr_range=[-10, 10]):
        self.logsnr_range = logsnr_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        logSNR = t * (self.logsnr_range[0]-self.logsnr_range[1]) + self.logsnr_range[1]
        return logSNR

class AdaptiveTrainSchedule(BaseSchedule):
    def __init__(self, logsnr_range=[-10, 10], buckets=100, min_probs=0.0):
        th = torch.linspace(logsnr_range[0], logsnr_range[1], buckets+1)
        self.bucket_ranges = torch.tensor([(th[i], th[i+1]) for i in range(buckets)])
        self.bucket_probs = torch.ones(buckets)
        self.min_probs = min_probs
        
    def schedule(self, t, batch_size):
        if t is not None:
            raise Exception("EDMTrainSchedule doesn't support passing timesteps: t")
        norm_probs = ((self.bucket_probs+self.min_probs) / (self.bucket_probs+self.min_probs).sum())
        buckets = torch.multinomial(norm_probs, batch_size, replacement=True)
        ranges = self.bucket_ranges[buckets]
        logSNR = torch.rand(batch_size) * (ranges[:, 1]-ranges[:, 0]) + ranges[:, 0]
        return logSNR

    def update_buckets(self, logSNR, loss, beta=0.99):
        range_mtx = self.bucket_ranges.unsqueeze(0).expand(logSNR.size(0), -1, -1).to(logSNR.device)
        range_mask = (range_mtx[:, :, 0] <= logSNR[:, None]) * (range_mtx[:, :, 1] > logSNR[:, None]).float()
        range_idx = range_mask.argmax(-1).cpu()
        self.bucket_probs[range_idx] = self.bucket_probs[range_idx] * beta + loss.cpu() * (1-beta)

class InterpolatedSchedule(BaseSchedule):
    def __init__(self, scheduler1, scheduler2, shifts=[1.0, 1.0]):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.shifts = shifts

    def schedule(self, t, batch_size):
        if t is None:
            t = 1-torch.rand(batch_size)
        low_logSNR = self.scheduler1(t, shift=self.shifts[0])
        high_logSNR = self.scheduler2(t, shift=self.shifts[1])
        return low_logSNR * t + high_logSNR * (1-t)
    
    