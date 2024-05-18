import numpy as np
import math

def p_init_Gauss(shape, dx, a, center, sigma2):
	p_init = np.zeros(shape)

	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				p_init[i, j, k] = a*math.exp(-((i*dx-center[0])**2+(j*dx-center[1])**2+(k*dx-center[2])**2)/2./sigma2)

	return p_init