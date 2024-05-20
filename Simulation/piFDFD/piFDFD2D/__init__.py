import numpy as np
import copy
from piFDFD.piFDFD2D import util, solver
import sys

"""
/********************/
2次元スカラーヘルムホルツ方程式の求解
/********************/
att:
	dx -> <float> 計算格子サイズ [m]
	shape -> <tuple:float:(2, )> 計算格子数 = (Nx, Ny)
	S -> <np:complex:(Nx, Ny)> B.C.込みの音源 = 連立一次方程式の右辺
	c -> <float> 音速 [m/s]
"""
class field:
	"""
	freq -> <float> 周波数 [Hz]
	voxel_label -> <np:int:(Nx, Ny)> 素材ラベル。空気はゼロ。
	beta -> <np:float:(max(voxel_label), )> 各ラベルに対応した比アドミッタンス値 (0 < beta < 1)
	"""
	def __init__(self, dx, freq, S, c = 340., voxel_label = None, beta = np.ones(1)):
		self.shape = S.shape
		self.dx = dx
		self.c = c
		self.k = 2.*np.pi*freq/self.c

		if voxel_label is None:
			voxel_label = np.ones(self.shape).astype(int)
			voxel_label[1:-1,1:-1] = 0
		self.voxel_label = voxel_label

		self.left_idxx, self.left_idxy, self.target_left_idxx, \
		self.right_idxx, self.right_idxy, self.target_right_idxx, \
		self.bottom_idxx, self.bottom_idxy, self.target_bottom_idxy, \
		self.top_idxx, self.top_idxy, self.target_top_idxy = util.voxel2mask(voxel_label)

		self.nanMask = np.zeros(self.shape).astype(bool)
		self.nanMask[voxel_label > 0] = True
		self.nanMask[self.left_idxx, self.left_idxy] = False
		self.nanMask[self.right_idxx, self.right_idxy] = False
		self.nanMask[self.bottom_idxx, self.bottom_idxy] = False
		self.nanMask[self.top_idxx, self.top_idxy] = False

		self.P = np.zeros(self.shape).astype(complex)
		self.P[self.nanMask] = np.nan + np.nan*1.j
		self.S = S
		self.S[self.nanMask] = np.nan + 1.j*np.nan

		self.beta_label = np.zeros(self.shape)
		for idx in range(len(beta)):
			self.beta_label[voxel_label == idx] = beta[idx]

	def getP(self, ghost = np.nan + 1.j*np.nan):
		P = copy.deepcopy(self.P)
		P[self.voxel_label > 0] = ghost

		return P[1:-1,1:-1]

	def mat(self, x):
		oput = (np.roll(x, 1, axis = 0) + np.roll(x, 1, axis = 1) + np.roll(x, -1, axis = 0) + np.roll(x, -1, axis = 1))/self.dx**2 + (self.k**2 - 4./self.dx**2)*x
		oput[self.left_idxx, self.left_idxy] = 0.
		oput[self.right_idxx, self.right_idxy] = 0.
		oput[self.top_idxx, self.top_idxy] = 0.
		oput[self.bottom_idxx, self.bottom_idxy] = 0.
		oput[self.left_idxx, self.left_idxy] += (1./self.dx - 1.j*self.k*self.beta_label[self.left_idxx, self.left_idxy]/2.)*x[self.left_idxx, self.left_idxy] - (1./self.dx + 1.j*self.k*self.beta_label[self.left_idxx, self.left_idxy]/2.)*x[self.target_left_idxx, self.left_idxy]
		oput[self.right_idxx, self.right_idxy] += (1./self.dx - 1.j*self.k*self.beta_label[self.right_idxx, self.right_idxy]/2.)*x[self.right_idxx, self.right_idxy] - (1./self.dx + 1.j*self.k*self.beta_label[self.right_idxx, self.right_idxy]/2.)*x[self.target_right_idxx, self.right_idxy]
		oput[self.top_idxx, self.top_idxy] += (1./self.dx - 1.j*self.k*self.beta_label[self.top_idxx, self.top_idxy]/2.)*x[self.top_idxx, self.top_idxy] - (1./self.dx + 1.j*self.k*self.beta_label[self.top_idxx, self.top_idxy]/2.)*x[self.top_idxx, self.target_top_idxy]
		oput[self.bottom_idxx, self.bottom_idxy] += (1./self.dx - 1.j*self.k*self.beta_label[self.bottom_idxx, self.bottom_idxy]/2.)*x[self.bottom_idxx, self.bottom_idxy] - (1./self.dx + 1.j*self.k*self.beta_label[self.bottom_idxx, self.bottom_idxy]/2.)*x[self.bottom_idxx, self.target_bottom_idxy]

		return oput

	def solve(self, iteration = 10000, epsilon = 1e-10, method = "BiCGSTABver1", s = 6):
		if method == "BiCGSTABver1":
			log = solver.BiCGSTABver1(self, iteration, epsilon)
		elif method == "IDR_s":
			log = solver.IDR_s(self, s, iteration, epsilon)
		else:
			pass

		return log

class field_continue(field):
	def __init__(self, F, freq):
		self.shape = F.shape
		self.dx = F.dx
		self.c = F.c
		self.k = 2.*np.pi*freq/F.c

		self.voxel_label = F.voxel_label
		self.left_idxx = F.left_idxx
		self.left_idxy = F.left_idxy
		self.target_left_idxx = F.target_left_idxx
		self.right_idxx = F.right_idxx
		self.right_idxy = F.right_idxy
		self.target_right_idxx = F.target_right_idxx
		self.bottom_idxx = F.bottom_idxx
		self.bottom_idxy = F.bottom_idxy
		self.target_bottom_idxy = F.target_bottom_idxy
		self.top_idxx = F.top_idxx
		self.top_idxy = F.top_idxy
		self.target_top_idxy = F.target_top_idxy

		self.nanMask = F.nanMask

		self.P = F.P
		self.S = F.S

		self.beta_label = F.beta_label