from SimulatedAnnealing.Temperature import util

class ECS:
    def __init__(self, T_init, alpha = 0.95):
        self.alpha = alpha
        self.T = T_init
    
    def __call__(self):
        return self.T
    
    def update(self):
        self.T *= self.alpha