import numpy as np
import math
from piFDTD.piWAVE2.util import p_init_Gauss
from piFDTD.piWAVE2.VTK import writeVTK

class Field:

	def __init__(self, p_init, dx, CFL = math.sqrt(0.5), c = 340., mask = None):
		self.dx = dx
		self.dt = CFL*(dx/c)

		size = p_init.shape
		if mask is None:
			mask = np.zeros(size).astype(bool)
		self.Mask = np.ones((size[0]+2, size[1]+2)).astype(bool)
		self.Mask[1:-1,1:-1] = mask

		self.P_after = np.zeros(size); self.P_after[self.Mask[1:-1, 1:-1]] = np.nan
		self.P = p_init; self.P[self.Mask[1:-1, 1:-1]] = np.nan
		self.P_before = p_init; self.P[self.Mask[1:-1, 1:-1]] = np.nan

	def getP(self, ghost = -100.):
		P = self.P.copy()
		P[self.Mask[1:-1,1:-1]] = ghost

		return P

	def update(self):
		for i in range(self.P.shape[0]):
			for j in range(self.P.shape[1]):
				if self.Mask[i+1, j+1] == False:
					if self.Mask[i+2, j+1]:
						Pip = self.P[i, j]
					else:
						Pip = self.P[i+1, j]

					if self.Mask[i, j+1]:
						Pim = self.P[i, j]
					else:
						Pim = self.P[i-1, j]

					if self.Mask[i+1, j+2]:
						Pjp = self.P[i, j]
					else:
						Pjp = self.P[i, j+1]

					if self.Mask[i+1, j]:
						Pjm = self.P[i, j]
					else:
						Pjm = self.P[i, j-1]

					self.P_after[i, j] = (Pip+Pim+Pjp+Pjm)/2. - self.P_before[i, j]

		self.P_before = self.P.copy()
		self.P = self.P_after.copy()