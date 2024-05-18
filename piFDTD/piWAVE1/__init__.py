import numpy as np
import math
from piFDTD.piWAVE1.util import p_init_Gauss

class Field:

	def __init__(self, p_init, dx, CFL = 1., c = 340.):
		size = len(p_init)
		self.dx = dx
		self.dt = CFL*(dx/c)

		self.P_after = np.zeros(p_init.shape)
		self.P = p_init
		self.P_before = p_init

	def update(self):
		self.P_after[1:-1] = self.P[2:]+self.P[:-2]-self.P_before[1:-1]
		self.P_after[0] = self.P[1]+ self.P[0] -self.P_before[0]
		self.P_after[-1] = self.P[-2]+ self.P[-1] -self.P_before[-1]

		self.P_before = self.P.copy()
		self.P = self.P_after.copy()