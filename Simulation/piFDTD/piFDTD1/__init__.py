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
from piFDTD.piFDTD1.util import p_init_Gauss

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

---p_initについて---
初期条件かつ-dtの圧力分布。
"""
class Field:

	def __init__(self, p_init, dx, CFL = 1., c = 340., rho = 1.293, k = 1.4e+5):
		size = len(p_init)
		self.dx = dx
		self.dt = CFL*(dx/c)
		self.rho = rho
		self.k = k

		self.P_after = p_init
		self.P_before = p_init
		self.U = np.zeros(size+1)

	def getP(self):
		P = (self.P_after + self.P_before)/2.

		return P

	def getVel(self):
		U_fix = (self.U[1:] + self.U[:-1])/2.

		return U_fix


	def update(self):
		self.P_before = self.P_after.copy()
		alpha = self.dt/self.dx/self.rho
		beta = self.dt*self.k/self.dx

		#####Uのupdate
		self.U[1:-1] -= alpha*(self.P_before[1:]-self.P_before[:-1])

		#####Pのupdate
		self.P_after = self.P_before - beta*(self.U[1:]-self.U[:-1])