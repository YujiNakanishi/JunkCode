import numpy as np
from SimulatedAnnealing import Sampling
from SimulatedAnnealing import Temperature
import copy

class Module:
    def __init__(self, x_init, sampler, T_scheduler, function):
        self.sampler = sampler
        self.T_scheduler = T_scheduler
        self.function = function

        self.x_history = [copy.deepcopy(x_init)]
        self.f_history = [function(x_init)]
    
    def step(self, num = 1):
        for n in range(num):
            x_new = self.sampler(self.x_history[-1])
            f_new = self.function(x_new)
            if f_new <= self.f_history[-1]:
                self.x_history.append(copy.deepcopy(x_new))
                self.f_history.append(f_new)
            else:
                if np.random.rand() < np.exp(-(f_new - self.f_history[-1])/self.T_scheduler()):
                    self.x_history.append(copy.deepcopy(x_new))
                    self.f_history.append(f_new)
                else:
                    self.x_history.append(copy.deepcopy(self.x_history[-1]))
                    self.f_history.append(self.f_history[-1])
            
            self.T_scheduler.update()
    
    def opt_solution(self):
        arg_min = np.argmin(self.f_history)
        return self.x_history[arg_min], self.f_history[arg_min]