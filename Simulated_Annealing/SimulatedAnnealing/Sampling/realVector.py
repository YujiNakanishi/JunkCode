import numpy as np

class Gauss:
    def __init__(self, sigma = 1.):
        self.sigma = sigma
    
    def __call__(self, x):
        return x + self.sigma*np.random.randn()