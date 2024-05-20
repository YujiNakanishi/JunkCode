import numpy as np
import math
from piFDTD.piWAVE3.util import p_init_Gauss
from piFDTD.piWAVE3.VTK import writeVTK

class Field:

	def __init__(self, p_init, dx, CFL = 1./math.sqrt(3.), c = 340., mask = None):
		self.dx = dx
		self.dt = CFL*(dx/c)

		size = p_init.shape
		if mask is None:
			mask = np.zeros(size).astype(bool)
		self.Mask = np.ones((size[0]+2, size[1]+2, size[2]+2)).astype(bool)
		self.Mask[1:-1,1:-1,1:-1] = mask

		self.P_after = np.zeros(size); self.P_after[self.Mask[1:-1, 1:-1, 1:-1]] = np.nan
		self.P = p_init; self.P[self.Mask[1:-1, 1:-1, 1:-1]] = np.nan
		self.P_before = p_init; self.P[self.Mask[1:-1, 1:-1, 1:-1]] = np.nan

	def getP(self, ghost = -100.):
		P = self.P.copy()
		P[self.Mask[1:-1,1:-1, 1:-1]] = ghost

		return P

	def update(self):
		for i in range(self.P.shape[0]):
			for j in range(self.P.shape[1]):
				for k in range(self.P.shape[2]):
					if self.Mask[i+1, j+1, k+1] == False:
						if self.Mask[i+2, j+1, k+1]:
							Pip = self.P[i, j, k]
						else:
							Pip = self.P[i+1, j, k]

						if self.Mask[i, j+1, k+1]:
							Pim = self.P[i, j, k]
						else:
							Pim = self.P[i-1, j, k]

						if self.Mask[i+1, j+2, k+1]:
							Pjp = self.P[i, j, k]
						else:
							Pjp = self.P[i, j+1, k]

						if self.Mask[i+1, j, k+1]:
							Pjm = self.P[i, j, k]
						else:
							Pjm = self.P[i, j-1, k]

						if self.Mask[i+1, j+1, k+2]:
							Pkp = self.P[i, j, k]
						else:
							Pkp = self.P[i, j, k+1]

						if self.Mask[i+1, j+1, k]:
							Pkm = self.P[i, j, k]
						else:
							Pkm = self.P[i, j, k-1]

						self.P_after[i, j, k] = (Pip+Pim+Pjp+Pjm+Pkp+Pkm)/3. - self.P_before[i, j, k]

		self.P_before = self.P.copy()
		self.P = self.P_after.copy()