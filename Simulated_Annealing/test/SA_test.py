import numpy as np
import SimulatedAnnealing as SA
from SimulatedAnnealing.Sampling.realVector import Gauss
from SimulatedAnnealing import Temperature as SAT

def function(x):
    return ((1. - x[0])**2 + 100.*(x[1]-x[0])**2)

objective_function = function
sampler = Gauss()
x_init = np.zeros(2)

T_init = SAT.util.decideInitialTemperature(objective_function, sampler, x_init, sample_num = 100)
T_scheduler = SAT.ECS(T_init)

module = SA.Module(x_init, sampler, T_scheduler, objective_function)
module.step(1000)

x_opt, f_opt = module.opt_solution()
print(x_opt)
print(f_opt)