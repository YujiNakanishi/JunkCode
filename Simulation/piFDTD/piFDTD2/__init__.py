"""
/**************************/
piFDTD
/**************************/
・2次元FDTD法ソルバー。
・境界条件はすべて固定端。
・インパルス応答のみSim.可能。したがって速度初期条件はゼロ。

---必要なモジュール---
numpy

---本ファイルで定義されている関数---
"""
import numpy as np
import math
from piFDTD.piFDTD2.util import p_init_Gauss
from piFDTD.piFDTD2.VTK import writeVTK

"""
/**********************/
Field
/**********************/
class : 様々な物理量を纏めたクラス。

---field---
dx -> <float> 格子刻み
dt -> <float> 時間刻み
P_after -> <np array> t+dt時の圧力分布。
P_before -> <np array> t-dt時の圧力分布。
U -> <np array> t時のx方向速度分布。
V -> <np array> t時のy方向速度分布。
Mask -> <np array> 固体 or 流体のマスク。if True -> 固体

---p_initについて---
初期条件かつ-dtの圧力分布。
"""
class Field:

	def __init__(self, p_init, dx, CFL = math.sqrt(0.5), c = 340., rho = 1.293, k = 1.4e+5, mask = None):
		size = p_init.shape
		self.dx = dx
		self.dt = CFL*(dx/c)
		self.rho = rho
		self.k = k

		self.P_after = p_init
		self.P_before = p_init
		self.U = np.zeros((size[0]+1, size[1]))
		self.V = np.zeros((size[0], size[1]+1))

		if mask is None:
			mask = np.zeros(size).astype(bool)
		self.Mask = np.ones((size[0]+2, size[1]+2)).astype(bool)
		self.Mask[1:-1,1:-1] = mask

	def getP(self, ghost_val = None):
		P = (self.P_after + self.P_before)/2.
		if not(ghost_val is None):
			P[self.Mask[1:-1,1:-1]] = ghost_val

		return P

	def getVel(self, ghost_val = None):
		U_fix = (self.U[1:,:] + self.U[:-1,:])/2.
		V_fix = (self.V[:,1:] + self.V[:,:-1])/2.

		if not(ghost_val is None):
			U_fix[self.Mask[1:-1,1:-1]] = ghost_val
			V_fix[self.Mask[1:-1,1:-1]] = ghost_val

		Vel = np.stack((U_fix, V_fix), axis = -1)

		return Vel


	def update(self):
		self.P_before = self.P_after.copy()
		alpha = self.dt/self.dx/self.rho
		beta = self.dt*self.k/self.dx

		#####Uのupdate
		for i in range(self.U.shape[0]):
			for j in range(self.U.shape[1]):
				if (self.Mask[i, j+1] == False) & (self.Mask[i+1, j+1] == False):
					self.U[i, j] -= alpha*(self.P_before[i, j]-self.P_before[i-1, j])

		#####Vのupdate
		for i in range(self.V.shape[0]):
			for j in range(self.V.shape[1]):
				if (self.Mask[i+1, j] == False) & (self.Mask[i+1, j+1] == False):
					self.V[i, j] -= alpha*(self.P_before[i, j]-self.P_before[i, j-1])

		#####Pのupdate
		for i in range(self.P_after.shape[0]):
			for j in range(self.P_after.shape[1]):
				if self.Mask[i+1, j+1] == False:
					self.P_after[i, j] = self.P_before[i, j] -beta*(self.U[i+1, j]-self.U[i, j]+self.V[i, j+1]-self.V[i, j])