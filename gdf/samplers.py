import torch

class SimpleSampler():
    def __init__(self, gdf):
        self.gdf = gdf
        self.current_step = -1

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        return torch.randn(*shape)

    def step(self, x, x0, epsilon, logSNR, logSNR_prev):
        raise NotImplementedError("You should override the 'apply' function.")

class DDIMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=0):
        a, b = self.gdf.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1]*(len(x0.shape)-1)), b.view(-1, *[1]*(len(x0.shape)-1))

        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1]*(len(x0.shape)-1)), b_prev.view(-1, *[1]*(len(x0.shape)-1))

        sigma_tau = eta * (b_prev**2 / b**2).sqrt() * (1 - a**2 / a_prev**2).sqrt() if eta > 0 else 0
        # x = a_prev * x0 + (1 - a_prev**2 - sigma_tau ** 2).sqrt() * epsilon + sigma_tau * torch.randn_like(x0)
        x = a_prev * x0 + (b_prev**2 - sigma_tau**2).sqrt() * epsilon + sigma_tau * torch.randn_like(x0)
        return x

class DDPMSampler(DDIMSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=1):
        return super().step(x, x0, epsilon, logSNR, logSNR_prev, eta)

class LCMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev):        
        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1]*(len(x0.shape)-1)), b_prev.view(-1, *[1]*(len(x0.shape)-1))
        return x0 * a_prev + torch.randn_like(epsilon) * b_prev

# --- RF Samplers


class RFSampler(BaseSampler):
    def step(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        logSNR: torch.Tensor,
        logSNR_prev: torch.Tensor,
        eta: float = 0,
    ) -> torch.Tensor:
        a, b = self.gdf.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1] * (len(x0.shape) - 1)), b.view(
                -1, *[1] * (len(x0.shape) - 1)
            )

        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1] * (len(x0.shape) - 1)), b_prev.view(
                -1, *[1] * (len(x0.shape) - 1)
            )

        return x0 * a_prev + epsilon * b_prev


class RF_DDIMSampler(BaseSampler):
    def step(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        logSNR: torch.Tensor,
        logSNR_prev: torch.Tensor,
        eta: float = 0,
    ) -> torch.Tensor:
        a, b = self.gdf.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1] * (len(x0.shape) - 1)), b.view(
                -1, *[1] * (len(x0.shape) - 1)
            )

        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1] * (len(x0.shape) - 1)), b_prev.view(
                -1, *[1] * (len(x0.shape) - 1)
            )

        # sigma_tau = eta * (s_prev / s) * (1 - (1-s) / (1-s_prev)) if eta > 0 else 0
        # s, s_prev = s.view(1)[:, None, None, None], s_prev.view(1)[:, None, None, None]
        # return (1-s_prev) * x0 + (s_prev - sigma_tau ** 2) * noise + sigma_tau * torch.randn_like(x0)

        sigma_tau = eta * (b_prev / b) * (1 - a / a_prev) if eta > 0 else 0
        x = (
            a_prev * x0
            + (b_prev - sigma_tau**2) * epsilon
            + sigma_tau * torch.randn_like(x0)
        )
        return x


class RF_DDPMSampler(RF_DDIMSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=1) -> torch.Tensor:
        return super().step(x, x0, epsilon, logSNR, logSNR_prev, eta)
    
