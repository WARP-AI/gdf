class VPScaler():
    def __call__(self, logSNR): 
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1-a_squared).sqrt()
        return a, b
    
class LERPScaler():
    def __call__(self, logSNR):
        _a = logSNR.exp() - 1
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        b = (1-a)
        return a, b