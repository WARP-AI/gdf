class BaseScaler():
    def __init__(self):
        self.stretched_limits = None

    def setup_limits(self, min_a, max_a, min_b, max_b):
        self.stretched_limits = min_a, max_a, min_b, max_b

    def stretch_limits(self, a, b):
        min_a, max_a, min_b, max_b = self.stretched_limits
        return a - min_a / (max_a - min_a), b - min_b / (max_b - min_b)
    
    def scalers(self, logSNR):
        raise Exception("this method needs to be overriden")
    
    def __call__(self, logSNR):
        a, b = self.scalers(logSNR)
        if self.stretched_limits is not None:
            a, b = self.stretch_limits(a, b)
        return a, b

class VPScaler(BaseScaler):
    def scalers(self, logSNR): 
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1-a_squared).sqrt()
        return a, b
    
class LERPScaler(BaseScaler):
    def scalers(self, logSNR):
        _a = logSNR.exp() - 1
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        b = (1-a)
        return a, b