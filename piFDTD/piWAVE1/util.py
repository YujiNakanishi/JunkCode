import numpy as np
import math

def p_init_Gauss(length, dx, a, center, sigma2):
	p_init = np.array([a*math.exp(-((i*dx-center)**2)/2./sigma2) for i in range(length)])

	return p_init