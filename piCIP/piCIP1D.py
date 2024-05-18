import numpy as np
import math


"""
/************************************/
CIP法による1次元音伝播シミュレーション
/************************************/
---att---
	Z -> <float> 特性インピーダンス
	dx -> <float> 格子刻み幅
	dt -> <float> 時間刻み
	epsilon -> <float> 係数 (陽に用いない)
	f, df -> <np:float:(N, )> 正方向伝播波およびその微分値
	g, dg -> <np:float:(N, )> 負方向伝播波およびその微分値
	r -> <tuple:float:(2, )> 左右の反射率。デフォルトは完全反射
"""
class CIP:
	def __init__(self, p_init, dp_init, dx, dt, r = (1., 1.), rho = 1.293, k = 1.4e+5):
		self.Z = math.sqrt(rho*k)
		self.dx = dx
		self.dt = dt
		self.epsilon = math.sqrt(k/rho)*self.dt
		self.f = np.copy(p_init); self.df = np.copy(dp_init)
		self.g = np.copy(p_init); self.dg = np.copy(dp_init)
		self.r = r

	def getP(self):
		return (self.f + self.g)/2.
	def getU(self):
		return (self.f - self.g)/(2.*self.Z)

	def boundaryCondition(self):
		self.f[0] = self.r[0]*self.g[0]; self.df[0] = -self.r[0]*self.dg[0]
		self.g[-1] = self.r[1]*self.f[-1]; self.dg[-1] = -self.r[1]*self.df[-1]

	def update(self):
		##########f, dfのupdate
		a = (self.df[1:]+self.df[:-1])/(self.dx**2) - 2.*(self.f[1:]-self.f[:-1])/(self.dx**3)
		b = 3.*(self.f[:-1]-self.f[1:])/(self.dx**2) + (2.*self.df[1:]+self.df[:-1])/self.dx

		self.f[1:] = -a*(self.epsilon**3) + b*(self.epsilon**2) - self.df[1:]*self.epsilon + self.f[1:]
		self.df[1:] = 3.*a*(self.epsilon**2) - 2.*b*self.epsilon + self.df[1:]

		##########g, dgのupdate
		a = (self.dg[:-1]+self.dg[1:])/(self.dx**2) + 2.*(self.g[:-1]-self.g[1:])/(self.dx**3)
		b = 3.*(self.g[1:]-self.g[:-1])/(self.dx**2) - (2.*self.dg[:-1]+self.dg[1:])/self.dx

		self.g[:-1] = a*(self.epsilon**3) + b*(self.epsilon**2) + self.dg[:-1]*self.epsilon + self.g[:-1]
		self.dg[:-1] = 3.*a*(self.epsilon**2) + 2.*b*self.epsilon + self.dg[:-1]

		##########境界条件
		self.boundaryCondition()


"""
/************************************/
CIP法による1次元音伝播シミュレーション：周波数依存B.C.(IIR表現)
/************************************/
---att---
<CIP>
	Z -> <float> 特性インピーダンス
	dx -> <float> 格子刻み幅
	dt -> <float> 時間刻み
	epsilon -> <float> 係数 (陽に用いない)
	f, df -> <np:float:(N, )> 正方向伝播波およびその微分値
	g, dg -> <np:float:(N, )> 負方向伝播波およびその微分値
<CIP_IIR>
	As -> <np:float:(D, )> 差分方程式の係数(DはIIRの次数+1)
	Bs -> <np:float:(D-1, )> 差分方程式の係数
"""
class CIP_IIR(CIP):
	def __init__(self, p_init, dp_init, dx, dt, As, Bs, rho = 1.293, k = 1.4e+5):
		self.Z = math.sqrt(rho*k)
		self.dx = dx
		self.dt = dt
		self.epsilon = math.sqrt(k/rho)*self.dt
		self.f = np.copy(p_init); self.df = np.copy(dp_init)
		self.g = np.copy(p_init); self.dg = np.copy(dp_init)
		self.As = As; self.Bs = Bs

		self.left_f_log = np.zeros(len(Bs))
		self.left_df_log = np.zeros(len(Bs))
		self.left_g_log = np.zeros(len(As))
		self.left_dg_log = np.zeros(len(As))

		self.right_g_log = np.zeros(len(Bs))
		self.right_dg_log = np.zeros(len(Bs))
		self.right_f_log = np.zeros(len(As))
		self.right_df_log = np.zeros(len(As))

	def boundaryCondition(self):
		#####左側更新
		self.left_g_log = np.roll(self.left_g_log, 1); self.left_g_log[0] = self.g[0]
		self.left_dg_log = np.roll(self.left_dg_log, 1); self.left_dg_log[0] = self.dg[0]

		self.f[0] = -np.sum(self.left_f_log * self.Bs) + np.sum(self.left_g_log * self.As)
		self.df[0] = -np.sum(self.left_df_log * self.Bs) - np.sum(self.left_dg_log * self.As)

		self.left_f_log = np.roll(self.left_f_log, 1); self.left_f_log[0] = self.f[0]
		self.left_df_log = np.roll(self.left_df_log, 1); self.left_df_log[0] = self.df[0]

		#####右側更新
		self.right_f_log = np.roll(self.right_f_log, 1); self.right_f_log[0] = self.f[-1]
		self.right_df_log = np.roll(self.right_df_log, 1); self.right_df_log[0] = self.df[-1]

		self.g[-1] = -np.sum(self.right_g_log * self.Bs) + np.sum(self.right_f_log * self.As)
		self.dg[-1] = -np.sum(self.right_dg_log * self.Bs) - np.sum(self.right_df_log * self.As)

		self.right_g_log = np.roll(self.right_g_log, 1); self.right_g_log[0] = self.g[-1]
		self.right_dg_log = np.roll(self.right_dg_log, 1); self.right_dg_log[0] = self.dg[-1]


"""
/************************************/
RCIP法による1次元音伝播シミュレーション
/************************************/
---att---
<CIP>
	Z -> <float> 特性インピーダンス
	dx -> <float> 格子刻み幅
	dt -> <float> 時間刻み
	epsilon -> <float> 係数 (陽に用いない)
	f, df -> <np:float:(N, )> 正方向伝播波およびその微分値
	g, dg -> <np:float:(N, )> 負方向伝播波およびその微分値
	r -> <tuple:float:(2, )> 左右の反射率。デフォルトは完全反射
<RCIP>
	alpha -> <float> 有理関数係数
"""
class RCIP(CIP):
	def __init__(self, p_init, dp_init, dx, dt, rho = 1.293, k = 1.4e+5, r = (1., 1.), alpha = 1.):
		super().__init__(p_init, dp_init, dx, dt, rho, k, r)
		self.alpha = alpha

	def update(self):
		##########f,dfのupdate
		S = (self.f[1:]-self.f[:-1])/self.dx
		B = (1-np.abs((S-self.df[1:])/(self.df[:-1]-S+1e-10)))/self.dx + 1e-10
		c = self.df[1:] + self.alpha*self.f[1:]*B
		a = (self.df[1:] - S + (self.df[:-1]-S)*(1-self.alpha*B*self.dx))/(self.dx**2)
		b = S*self.alpha*B - (S-self.df[1:])/self.dx + a*self.dx

		self.f[1:] = (-a*(self.epsilon**3) + b*(self.epsilon**2) -c*self.epsilon + self.f[1:])/(1.-self.alpha*B*self.epsilon)
		self.df[1:] = (3.*a*(self.epsilon**2) - 2.*b*self.epsilon + c)/(1.-self.alpha*B*self.epsilon) \
		-self.alpha*B*self.f[1:]/(1.-self.alpha*B*self.epsilon)

		# ##########g, dgのupdate
		S = (self.g[1:]-self.g[:-1])/self.dx
		B = (np.abs((S-self.dg[:-1])/(self.dg[1:]-S+1e-10))-1.)/self.dx + 1e-10
		c = self.dg[:-1] + self.g[:-1]*self.alpha*B
		a = (self.dg[:-1]-S+(self.dg[1:]-S)*(1.+self.alpha*B*self.dx))/(self.dx**2)
		b = S*self.alpha*B + (S-self.dg[:-1])/self.dx - a*self.dx

		self.g[:-1] = (a*(self.epsilon**3)+b*(self.epsilon**2)+c*self.epsilon+self.g[:-1])/(1.+self.alpha*B*self.epsilon)
		self.dg[:-1] = (3.*a*(self.epsilon**2) + 2.*b*self.epsilon + c)/(1.+self.alpha*B*self.epsilon) \
		-self.alpha*B*self.g[:-1]/(1.+self.alpha*B*self.epsilon)

		##########境界条件
		self.boundaryCondition()


"""
/************************************/
RCIP法による1次元音伝播シミュレーション：周波数依存B.C.(IIR表現)
/************************************/
---att---
---att---
<CIP>
	Z -> <float> 特性インピーダンス
	dx -> <float> 格子刻み幅
	dt -> <float> 時間刻み
	epsilon -> <float> 係数 (陽に用いない)
	f, df -> <np:float:(N, )> 正方向伝播波およびその微分値
	g, dg -> <np:float:(N, )> 負方向伝播波およびその微分値
<CIP_IIR>
	As -> <np:float:(D, )> 差分方程式の係数(DはIIRの次数+1)
	Bs -> <np:float:(D-1, )> 差分方程式の係数
<RCIP_IIR>
	alpha -> <float> 有理関数係数
"""
class RCIP_IIR(CIP_IIR):
	def __init__(self, p_init, dp_init, dx, dt, As, Bs, rho = 1.293, k = 1.4e+5, alpha = 1.):
		super().__init__(p_init, dp_init, dx, dt, As, Bs, rho, k)
		self.alpha = alpha

	def update(self):
		##########f,dfのupdate
		S = (self.f[1:]-self.f[:-1])/self.dx
		B = (1-np.abs((S-self.df[1:])/(self.df[:-1]-S+1e-10)))/self.dx + 1e-10
		c = self.df[1:] + self.alpha*self.f[1:]*B
		a = (self.df[1:] - S + (self.df[:-1]-S)*(1-self.alpha*B*self.dx))/(self.dx**2)
		b = S*self.alpha*B - (S-self.df[1:])/self.dx + a*self.dx

		self.f[1:] = (-a*(self.epsilon**3) + b*(self.epsilon**2) -c*self.epsilon + self.f[1:])/(1.-self.alpha*B*self.epsilon)
		self.df[1:] = (3.*a*(self.epsilon**2) - 2.*b*self.epsilon + c)/(1.-self.alpha*B*self.epsilon) \
		-self.alpha*B*self.f[1:]/(1.-self.alpha*B*self.epsilon)

		# ##########g, dgのupdate
		S = (self.g[1:]-self.g[:-1])/self.dx
		B = (np.abs((S-self.dg[:-1])/(self.dg[1:]-S+1e-10))-1.)/self.dx + 1e-10
		c = self.dg[:-1] + self.g[:-1]*self.alpha*B
		a = (self.dg[:-1]-S+(self.dg[1:]-S)*(1.+self.alpha*B*self.dx))/(self.dx**2)
		b = S*self.alpha*B + (S-self.dg[:-1])/self.dx - a*self.dx

		self.g[:-1] = (a*(self.epsilon**3)+b*(self.epsilon**2)+c*self.epsilon+self.g[:-1])/(1.+self.alpha*B*self.epsilon)
		self.dg[:-1] = (3.*a*(self.epsilon**2) + 2.*b*self.epsilon + c)/(1.+self.alpha*B*self.epsilon) \
		-self.alpha*B*self.g[:-1]/(1.+self.alpha*B*self.epsilon)

		##########境界条件
		self.boundaryCondition()