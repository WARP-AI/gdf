class EpsilonTarget():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon

    def x0_eps(self, noised, pred, logSNR, a, b):
        return (noised - pred * b) / a, pred # x0, epsilon
    
class X0Target():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return x0

    def x0_eps(self, noised, pred, logSNR, a, b):
        return pred, (noised - pred * a) / b # x0, epsilon

class VTarget():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return a * epsilon - b * x0

    def x0_eps(self, noised, pred, logSNR, a, b):
        squared_sum = (a**2 + b**2)
        return a/squared_sum * noised - b/squared_sum * pred, b/squared_sum * noised + a/squared_sum * pred

class RectifiedFlowsTarget():
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon - x0

    def x0_eps(self, noised, pred, logSNR, a, b):
        return noised - pred * b, noised + pred * a