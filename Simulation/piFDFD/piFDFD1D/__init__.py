import numpy as np
from piFDFD.piFDFD1D import solver

"""
/********************/
1次元スカラーヘルムホルツ方程式の求解
/********************/
att:
	N -> <int> 計算格子数 [-]
	dx -> <float> 格子サイズ [m]
	k -> <np:float:(N, )> 波数 [1/m]
	S -> <np:complex:(N, )> B.C.寄与込みのソース項 = 連立一次方程式右辺
	P -> <np:complex:(N, )> 複素音圧
	beta_left, beta_right -> <float> 比アドミッタンス [-] 0 < beta < 1
"""
class field:
	"""
	input:
		freq -> <float> 周波数 [Hz]
		c -> <float> 音速 [m/s]
	"""
	def __init__(self, dx, freq, S, beta_left = 1., beta_right = 1., c = 340.):
		self.N = len(S)
		self.dx = dx
		self.c = c
		self.k = 2.*np.pi*freq/c

		self.S = S
		self.P = np.zeros(self.N).astype(complex)
		self.beta_left = beta_left; self.beta_right = beta_right

	def getP(self):
		return self.P[1:-1]

	"""
	/******************/
	係数行列の作用。陽には用いない。
	/******************/
	input : x -> <np:complex:(N, )>
	"""
	def mat(self, x):
		oput = np.zeros(x.shape).astype(complex)
		oput[1:-1] = (x[2:] + x[:-2])/(self.dx**2) + (self.k**2 - 2./self.dx**2)*x[1:-1]
		oput[0] = (1./self.dx - 1.j*self.k*self.beta_left/2.)*x[0] - (1./self.dx + 1.j*self.k*self.beta_left/2.)*x[1]
		oput[-1] = (1./self.dx - 1.j*self.k*self.beta_right/2.)*x[-1] - (1./self.dx + 1.j*self.k*self.beta_right/2.)*x[-2]

		return oput

	"""
	/******************/
	定常計算。
	/******************/
	input :
		iteration -> <int> 最大反復回数
		epsilon -> <float> 最大許容残差
		solver -> <str> 連立一次方程式求解ソルバー
	output :
		log -> <np:float:(itr, )> 残差の系列
	"""
	def solve(self, iteration = 10000, epsilon = 1e-10, method = "BiCGSTABver1"):
		if method == "BiCGSTABver1":
			log = solver.BiCGSTABver1(self, iteration, epsilon)
		else:
			pass

		return log

"""
/********************/
1次元スカラーヘルムホルツ方程式の求解。初期近似解の転用。
/********************/
"""
class field_continue(field):
	"""
	F -> <class:field>
	freq -> <float> 周波数 [Hz]
	"""
	def __init__(self, F, freq):
		self.N = F.N
		self.dx = F.dx
		self.c = F.c
		self.k = 2.*np.pi*freq/self.c

		self.S = F.S
		self.P = F.P
		self.beta_left = F.beta_left; self.beta_right = F.beta_right


"""
/********************/
1次元スカラーヘルムホルツ方程式の求解。ガウスの消去法による直接解法
/********************/
att:
	A -> <np:complex:(N, N)> 係数行列
"""
class field_direct(field):
	def __init__(self, dx, freq, S, beta_left = 1., beta_right = 1., c = 340.):
		super().__init__(dx, freq, S, beta_left, beta_right, c)

		self.A = np.zeros((self.N, self.N)).astype(complex)
		self.A[0,0] = 1./self.dx - 1.j*self.k*self.beta_left/2.
		self.A[0,1] = -1./self.dx - 1.j*self.k*self.beta_left/2.
		self.A[-1,-1] = 1./self.dx - 1.j*self.k*self.beta_right/2.
		self.A[-1,-2] = -1./self.dx - 1.j*self.k*self.beta_right/2.

		for i in range(1, self.N-1):
			self.A[i,i] = (self.k**2 - 2/dx**2)
			self.A[i,i+1] = 1/self.dx**2
			self.A[i,i-1] = 1/self.dx**2

	def solve(self):
		self.P = np.linalg.solve(self.A, self.S)