def BaseScaler():
    def __init__(self):
        self.adjusted_limits = None

    def schedule(self, logSNR):
        raise Exception("this method needs to be overriden")

    def setup_limits(self, min_a, max_a, min_b, max_b):
        self.adjusted_limits = min_a, max_a, min_b, max_b

    def adjust_limits(self, a, b):
        min_a, max_a, min_b, max_b = self.adjusted_limits
        return a - min_a / (max_a - min_a), b - min_b / (max_b - min_b)

class VPScaler(BaseScaler):
    def __call__(self, logSNR): 
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1-a_squared).sqrt()
        if self.adjusted_limits is not None:
            a, b = self.adjust_limits(a, b)
        return a, b
    
class LERPScaler(BaseScaler):
    def __call__(self, logSNR):
        _a = logSNR.exp() - 1
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        b = (1-a)
        if self.adjusted_limits is not None:
            a, b = self.adjust_limits(a, b)
        return a, b