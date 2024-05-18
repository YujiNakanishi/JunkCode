import numpy as np
import math
from piCIP.piCIP2D import util

"""
/*******************************/
M型CIP法による2次元音伝播シミュレーション
/*******************************/
"""
class MCIP:
	"""
	/*****************************/
	process : コンストラクタ
	/*****************************/
	input:
		p_init, dxp_init, dyp_init -> <np:float:(X, Y)> 圧力初期分布
		dx, dt -> <float> 刻み幅
		voxel_label -> <np:int:(X+2, Y+2)> 計算格子材料物性ラベル。
		Rs -> <np:float:(max(voxel_label), )> 各材質(ラベル)の反射率。
	Note:
	---voxel_label---
	空気のラベルはゼロ。
	"""
	def __init__(self, p_init, dxp_init, dyp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = np.ones(1)):
		shape = (p_init.shape[0]+2, p_init.shape[1]+2)
		
		self.Z = math.sqrt(rho*k)
		self.dx = dx
		self.dt = dt
		self.epsilon = math.sqrt(k/rho)*self.dt

		if voxel_label is None:
			voxel_label = np.ones(shape); voxel_label[1:-1,1:-1] = 0
		assert np.max(voxel_label) == len(Rs), "voxel_label error"

		self.fx_mask, self.gx_mask, self.fy_mask, self.gy_mask, self.wall_mask, self.r_fx, self.r_gx, self.r_fy, self.r_gy = util.Label2Mask(voxel_label, Rs)
		
		self.P = np.zeros(shape); self.P[1:-1,1:-1] = p_init; self.P[self.wall_mask] = np.nan
		self.dxP = np.zeros(shape); self.dxP[1:-1,1:-1] = dxp_init; self.dxP[self.wall_mask] = np.nan
		self.dyP = np.zeros(shape); self.dyP[1:-1,1:-1] = dyp_init; self.dyP[self.wall_mask] = np.nan

		self.U = np.zeros(shape); self.U[self.wall_mask] = np.nan
		self.V = np.zeros(shape); self.V[self.wall_mask] = np.nan
		self.dxU = np.zeros(shape); self.dxU[self.wall_mask] = np.nan
		self.dyU = np.zeros(shape); self.dyU[self.wall_mask] = np.nan
		self.dxV = np.zeros(shape); self.dxV[self.wall_mask] = np.nan
		self.dyV = np.zeros(shape); self.dyV[self.wall_mask] = np.nan

		self.static = 0

	def getP(self, ghost_val = None):
		P = np.copy(self.P)
		if not(ghost_val is None):
			P[self.wall_mask] = ghost_val
		return P[1:-1,1:-1]

	def getVel(self, ghost_val = None):
		U = np.copy(self.U); V = np.copy(self.V)
		if not(ghost_val is None):
			U[self.wall_mask] = ghost_val; V[self.wall_mask] = ghost_val
		Vel = np.stack((U[1:-1,1:-1], V[1:-1,1:-1]), axis = -1)
		return Vel

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		a = (df + dfup)/(D**2) + 2.*(f-fup)/(D**3)
		b = 3.*(fup-f)/(D**2) - (2.*df+dfup)/D

		f_new = a*(epsilon**3) + b*(epsilon**2) + df*epsilon + f
		df_new = 3.*a*(epsilon**2) + 2.*b*epsilon + df

		return f_new, df_new

	def boundaryconditionx(self, fi, gi, difi, digi, djfi, djgi):
		fi[self.fx_mask] = gi[self.fx_mask] * self.r_fx
		difi[self.fx_mask] = -digi[self.fx_mask] * self.r_fx
		djfi[self.fx_mask] = djgi[self.fx_mask] * self.r_fx

		gi[self.gx_mask] = fi[self.gx_mask] * self.r_gx
		digi[self.gx_mask] = -difi[self.gx_mask] * self.r_gx
		djgi[self.gx_mask] = djfi[self.gx_mask] * self.r_gx

		return fi, gi, difi, digi, djfi, djgi

	def boundaryconditiony(self, fi, gi, difi, digi, djfi, djgi):
		fi[self.fy_mask] = gi[self.fy_mask] * self.r_fy
		difi[self.fy_mask] = -digi[self.fy_mask] * self.r_fy
		djfi[self.fy_mask] = djgi[self.fy_mask] * self.r_fy

		gi[self.gy_mask] = fi[self.gy_mask] * self.r_gy
		digi[self.gy_mask] = -difi[self.gy_mask] * self.r_gy
		djgi[self.gy_mask] = djfi[self.gy_mask] * self.r_gy

		return fi, gi, difi, digi, djfi, djgi

	def updatex(self):
		##########x方向の移流
		#####fx, dxfx, dyfxの定義
		fx = self.P + self.Z*self.U; dxfx = self.dxP+self.Z*self.dxU; dyfx = self.dyP+self.Z*self.dyU
		#####fx, dxfxの更新
		fx[1:,:], dxfx[1:,:] = self.interpolate(fx[1:,:], dxfx[1:,:], fx[:-1,:], dxfx[:-1,:], -self.dx, -self.epsilon)

		#####dyfxの更新
		dyfx[1:,:] = (1.-self.epsilon/self.dx)*dyfx[1:,:] + (self.epsilon/self.dx)*dyfx[:-1,:]

		#####gx, dxgx, dygxの定義
		gx = self.P-self.Z*self.U; dxgx = self.dxP-self.Z*self.dxU; dygx = self.dyP-self.Z*self.dyU
		#####gx, dxgxの更新
		gx[:-1,:], dxgx[:-1,:] = self.interpolate(gx[:-1,:], dxgx[:-1,:], gx[1:,:], dxgx[1:,:], self.dx, self.epsilon)

		#####dygxの更新
		dygx[:-1,:] = (1. - self.epsilon/self.dx)*dygx[:-1,:] + (self.epsilon/self.dx)*dygx[1:,:]

		#####境界条件
		fx, gx, dxfx, dxgx, dyfx, dygx = self.boundaryconditionx(fx, gx, dxfx, dxgx, dyfx, dygx)

		##########物理量の更新
		self.P = (fx + gx)/2.; self.P[self.wall_mask] = np.nan
		self.U = (fx - gx)/(2.*self.Z); self.U[self.wall_mask] = np.nan 
		self.dxP = (dxfx + dxgx)/2.; self.dxP[self.wall_mask] = np.nan
		self.dxU = (dxfx - dxgx)/(2.*self.Z); self.dxU[self.wall_mask] = np.nan
		self.dyP = (dyfx + dygx)/2.; self.dyP[self.wall_mask] = np.nan
		self.dyU = (dyfx - dygx)/(2.*self.Z); self.dyU[self.wall_mask] = np.nan

	def updatey(self):
		##########y方向の移流
		#####fy, dxfy, dyfyの定義
		fy = self.P + self.Z*self.V; dxfy = self.dxP+self.Z*self.dxV; dyfy = self.dyP+self.Z*self.dyV
		#####fy, dyfyの更新
		fy[:,1:], dyfy[:,1:] = self.interpolate(fy[:,1:], dyfy[:,1:], fy[:,:-1], dyfy[:,:-1], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfy[:,1:] = (1.-self.epsilon/self.dx)*dxfy[:,1:] + (self.epsilon/self.dx)*dxfy[:,:-1]

		#####gy, dxgy, dygyの定義
		gy = self.P-self.Z*self.V; dxgy = self.dxP-self.Z*self.dxV; dygy = self.dyP-self.Z*self.dyV
		#####gy, dygyの更新
		gy[:,:-1], dygy[:,:-1] = self.interpolate(gy[:,:-1], dygy[:,:-1], gy[:,1:], dygy[:,1:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgy[:,:-1] = (1.-self.epsilon/self.dx)*dxgy[:,:-1] + (self.epsilon/self.dx)*dxgy[:,1:]

		#####境界条件
		fy, gy, dyfy, dygy, dxfy, dxgy = self.boundaryconditiony(fy, gy, dyfy, dygy, dxfy, dxgy)

		##########物理量の更新
		self.P = (fy + gy)/2.; self.P[self.wall_mask] = np.nan
		self.V = (fy - gy)/(2.*self.Z); self.V[self.wall_mask] = np.nan
		self.dyP = (dyfy + dygy)/2.; self.dyP[self.wall_mask] = np.nan
		self.dyV = (dyfy - dygy)/(2.*self.Z); self.dyV[self.wall_mask] = np.nan
		self.dxP = (dxfy + dxgy)/2.; self.dxP[self.wall_mask] = np.nan
		self.dxV = (dxfy - dxgy)/(2.*self.Z); self.dxV[self.wall_mask] = np.nan

	def update(self):
		if (self.static%2) == 0:
			self.updatex(); self.updatey()
		else:
			self.updatey(); self.updatex()

		self.static += 1


"""
/*******************************/
M型RCIP法による2次元音伝播シミュレーション
/*******************************/
"""
class MRCIP(MCIP):
	"""
	/*****************************/
	process : コンストラクタ
	/*****************************/
	input:
		alpha -> <float> 有理関数の係数
	"""
	def __init__(self, p_init, dxp_init, dyp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = np.ones(1), alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dx, dt, rho, k, voxel_label, Rs)
		self.alpha = alpha

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (np.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new


"""
/************************************/
C型CIP法による2次元音伝播シミュレーション
/************************************/
"""
class CCIP(MCIP):
	def __init__(self, p_init, dxp_init, dyp_init, dxyp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = np.ones(1)):
		super().__init__(p_init, dxp_init, dyp_init, dx, dt, rho, k, voxel_label, Rs)
		shape = (p_init.shape[0]+2, p_init.shape[1]+2)
		self.dxyP = np.zeros(shape); self.dxyP[1:-1,1:-1] = dxyp_init; self.dxyP[self.wall_mask] = np.nan
		self.dxyU = np.zeros(shape); self.dxyU[self.wall_mask] = np.nan
		self.dxyV = np.zeros(shape); self.dxyV[self.wall_mask] = np.nan

	def boundaryconditionx(self, fi, gi, difi, digi, djfi, djgi, dijfi, dijgi):
		fi[self.fx_mask] = gi[self.fx_mask] * self.r_fx
		difi[self.fx_mask] = -digi[self.fx_mask] * self.r_fx
		djfi[self.fx_mask] = djgi[self.fx_mask] * self.r_fx
		dijfi[self.fx_mask] = -dijgi[self.fx_mask] * self.r_fx

		gi[self.gx_mask] = fi[self.gx_mask] * self.r_gx
		digi[self.gx_mask] = -difi[self.gx_mask] * self.r_gx
		djgi[self.gx_mask] = djfi[self.gx_mask] * self.r_gx
		dijgi[self.gx_mask] = -dijfi[self.gx_mask] * self.r_gx

		return fi, gi, difi, digi, djfi, djgi, dijfi, dijgi

	def boundaryconditiony(self, fi, gi, difi, digi, djfi, djgi, dijfi, dijgi):
		fi[self.fy_mask] = gi[self.fy_mask] * self.r_fy
		difi[self.fy_mask] = -digi[self.fy_mask] * self.r_fy
		djfi[self.fy_mask] = djgi[self.fy_mask] * self.r_fy
		dijfi[self.fy_mask] = -dijgi[self.fy_mask] * self.r_fy

		gi[self.gy_mask] = fi[self.gy_mask] * self.r_gy
		digi[self.gy_mask] = -difi[self.gy_mask] * self.r_gy
		djgi[self.gy_mask] = djfi[self.gy_mask] * self.r_gy
		dijgi[self.gy_mask] = -dijfi[self.gy_mask] * self.r_gy

		return fi, gi, difi, digi, djfi, djgi, dijfi, dijgi

	def updatex(self):
		##########x方向の移流
		#####fx, dxfx, dyfx, dxyfxの定義
		fx = self.P + self.Z*self.U
		dxfx = self.dxP + self.Z*self.dxU
		dyfx = self.dyP + self.Z*self.dyU
		dxyfx = self.dxyP + self.Z*self.dxyU 
		#####fx, dxfxの更新
		fx[1:,:], dxfx[1:,:] = self.interpolate(fx[1:,:], dxfx[1:,:], fx[:-1,:], dxfx[:-1,:], -self.dx, -self.epsilon)
		#####dyfx, dxyfxの更新
		dyfx[1:,:], dxyfx[1:,:] = self.interpolate(dyfx[1:,:], dxyfx[1:,:], dyfx[:-1,:], dxyfx[:-1,:], -self.dx, -self.epsilon)

		#####gx, dxgx, dygx, dxygxの定義
		gx = self.P - self.Z*self.U
		dxgx = self.dxP - self.Z*self.dxU
		dygx = self.dyP - self.Z*self.dyU
		dxygx = self.dxyP - self.Z*self.dxyU
		#####gx, dxgxの更新
		gx[:-1,:], dxgx[:-1,:] = self.interpolate(gx[:-1,:], dxgx[:-1,:], gx[1:,:], dxgx[1:,:], self.dx, self.epsilon)
		#####dygxの更新
		dygx[:-1,:], dxygx[:-1,:] = self.interpolate(dygx[:-1,:], dxygx[:-1,:], dygx[1:,:], dxygx[1:,:], self.dx, self.epsilon)

		#####境界条件
		fx, gx, dxfx, dxgx, dyfx, dygx, dxyfx, dxygx = self.boundaryconditionx(fx, gx, dxfx, dxgx, dyfx, dygx, dxyfx, dxygx)

		##########物理量の更新
		self.P = (fx + gx)/2.; self.P[self.wall_mask] = np.nan 
		self.U = (fx - gx)/(2.*self.Z); self.U[self.wall_mask] = np.nan
		self.dxP = (dxfx + dxgx)/2.; self.dxP[self.wall_mask] = np.nan
		self.dxU = (dxfx - dxgx)/(2.*self.Z); self.dxU[self.wall_mask] = np.nan
		self.dyP = (dyfx + dygx)/2.; self.dyP[self.wall_mask] = np.nan
		self.dyU = (dyfx - dygx)/(2.*self.Z); self.dyU[self.wall_mask] = np.nan
		self.dxyP = (dxyfx + dxygx)/2.; self.dxyP[self.wall_mask] = np.nan
		self.dxyU = (dxyfx - dxygx)/(2.*self.Z); self.dxyU[self.wall_mask] = np.nan

	def updatey(self):
		##########y方向の移流
		#####fy, dxfy, dyfy, dxyfyの定義
		fy = self.P + self.Z*self.V
		dxfy = self.dxP + self.Z*self.dxV
		dyfy = self.dyP + self.Z*self.dyV
		dxyfy = self.dxyP + self.Z*self.dxyV
		#####fy, dyfyの更新
		fy[:,1:], dyfy[:,1:] = self.interpolate(fy[:,1:], dyfy[:,1:], fy[:,:-1], dyfy[:,:-1], -self.dx, -self.epsilon)
		#####dxfyの更新
		dxfy[:,1:], dxyfy[:,1:] = self.interpolate(dxfy[:,1:], dxyfy[:,1:], dxfy[:,:-1], dxyfy[:,:-1], -self.dx, -self.epsilon)

		#####gy, dxgy, dygyの定義
		gy = self.P-self.Z*self.V
		dxgy = self.dxP-self.Z*self.dxV
		dygy = self.dyP-self.Z*self.dyV
		dxygy = self.dxyP-self.Z*self.dxyV
		
		#####gy, dygyの更新
		gy[:,:-1], dygy[:,:-1] = self.interpolate(gy[:,:-1], dygy[:,:-1], gy[:,1:], dygy[:,1:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgy[:,:-1], dxygy[:,:-1] = self.interpolate(dxgy[:,:-1], dxygy[:,:-1], dxgy[:,1:], dxygy[:,1:], self.dx, self.epsilon)

		#####境界条件
		fy, gy, dyfy, dygy, dxfy, dxgy, dxyfy, dxygy = self.boundaryconditiony(fy, gy, dyfy, dygy, dxfy, dxgy, dxyfy, dxygy)

		##########物理量の更新
		self.P = (fy + gy)/2.; self.P[self.wall_mask] = np.nan
		self.V = (fy - gy)/(2.*self.Z); self.V[self.wall_mask] = np.nan
		self.dyP = (dyfy + dygy)/2.; self.dyP[self.wall_mask] = np.nan
		self.dyV = (dyfy - dygy)/(2.*self.Z); self.dyV[self.wall_mask] = np.nan
		self.dxP = (dxfy + dxgy)/2.; self.dxP[self.wall_mask] = np.nan
		self.dxV = (dxfy - dxgy)/(2.*self.Z); self.dyV[self.wall_mask] = np.nan
		self.dxyP = (dxyfy + dxygy)/2.; self.dxyP[self.wall_mask] = np.nan
		self.dxyV = (dxyfy - dxygy)/(2.*self.Z); self.dxyV[self.wall_mask] = np.nan

"""
/************************************/
C型RCIP法による2次元音伝播シミュレーション
/************************************/
"""
class CRCIP(CCIP):
	def __init__(self, p_init, dxp_init, dyp_init, dxyp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = np.ones(1), alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dxyp_init, dx, dt, rho, k, voxel_label, Rs)
		self.alpha = alpha

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (np.abs((S-df)/(dfup-S+1e-10))-1.)/D
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new

"""
/******************************/
M型CIP法による2次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class MCIP_IIR(MCIP):
	"""
	/*********************/
	process : コンストラクタ
	/*********************/
	input:
		As -> <np:float:(max(voxel_label), order)> IIRフィルタ係数
		Bs -> <np:float:(max(voxel_label), order-1)> IIRフィルタ係数
	"""
	def __init__(self, p_init, dxp_init, dyp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5):
		assert np.max(voxel_label) == len(As), "voxel_label error"
		order = As.shape[1]
		shape = (p_init.shape[0]+2, p_init.shape[1]+2)
		
		self.Z = math.sqrt(rho*k)
		self.dx = dx
		self.dt = dt
		self.epsilon = math.sqrt(k/rho)*self.dt

		self.fx_mask, self.gx_mask, self.fy_mask, self.gy_mask, self.wall_mask, self.As_fx, self.Bs_fx, self.As_gx, self.Bs_gx, self.As_fy, self.Bs_fy, self.As_gy, self.Bs_gy = util.Label2Mask_IIR(voxel_label, As, Bs)

		self.P = np.zeros(shape); self.P[1:-1,1:-1] = p_init; self.P[self.wall_mask] = np.nan
		self.dxP = np.zeros(shape); self.dxP[1:-1,1:-1] = dxp_init; self.dxP[self.wall_mask] = np.nan
		self.dyP = np.zeros(shape); self.dyP[1:-1,1:-1] = dyp_init; self.dyP[self.wall_mask] = np.nan

		self.U = np.zeros(shape); self.U[self.wall_mask] = np.nan
		self.V = np.zeros(shape); self.V[self.wall_mask] = np.nan
		self.dxU = np.zeros(shape); self.dxU[self.wall_mask] = np.nan
		self.dyU = np.zeros(shape); self.dyU[self.wall_mask] = np.nan
		self.dxV = np.zeros(shape); self.dxV[self.wall_mask] = np.nan
		self.dyV = np.zeros(shape); self.dyV[self.wall_mask] = np.nan

		self.log_fx_out = np.zeros((order-1, len(self.fx_mask[0])))
		self.log_fx_out[0,:] = self.P[self.fx_mask] + self.Z*self.U[self.fx_mask]
		self.log_gx_in = np.zeros((order, len(self.fx_mask[0])))
		self.log_gx_in[0,:] = self.P[self.fx_mask] - self.Z*self.U[self.fx_mask]
		self.log_dxfx_out = np.zeros((order-1, len(self.fx_mask[0])))
		self.log_dxfx_out[0,:] = self.dxP[self.fx_mask] + self.Z*self.dxU[self.fx_mask]
		self.log_dxgx_in = np.zeros((order, len(self.fx_mask[0])))
		self.log_dxgx_in[0,:] = self.dxP[self.fx_mask] - self.Z*self.dxU[self.fx_mask]
		self.log_dyfx_out = np.zeros((order-1, len(self.fx_mask[0])))
		self.log_dyfx_out[0,:] = self.dyP[self.fx_mask] + self.Z*self.dyU[self.fx_mask]
		self.log_dygx_in = np.zeros((order, len(self.fx_mask[0])))
		self.log_dygx_in[0,:] = self.dyP[self.fx_mask] - self.Z*self.dyU[self.fx_mask]

		self.log_gx_out = np.zeros((order-1, len(self.gx_mask[0])))
		self.log_gx_out[0,:] = self.P[self.gx_mask] - self.Z*self.U[self.gx_mask]
		self.log_fx_in = np.zeros((order, len(self.gx_mask[0])))
		self.log_fx_in[0,:] = self.P[self.gx_mask] + self.Z*self.U[self.gx_mask]
		self.log_dxgx_out = np.zeros((order-1, len(self.gx_mask[0])))
		self.log_dxgx_out[0,:] = self.dxP[self.gx_mask] - self.Z*self.dxU[self.gx_mask]
		self.log_dxfx_in = np.zeros((order, len(self.gx_mask[0])))
		self.log_dxfx_in[0,:] = self.dxP[self.gx_mask] + self.Z*self.dxU[self.gx_mask]
		self.log_dygx_out = np.zeros((order-1, len(self.gx_mask[0])))
		self.log_dygx_out[0,:] = self.dyP[self.gx_mask] - self.Z*self.dyU[self.gx_mask]
		self.log_dyfx_in = np.zeros((order, len(self.gx_mask[0])))
		self.log_dyfx_in[0,:] = self.dyP[self.gx_mask] + self.Z*self.dyU[self.gx_mask]

		self.log_fy_out = np.zeros((order-1, len(self.fy_mask[0])))
		self.log_fy_out[0,:] = self.P[self.fy_mask] + self.Z*self.V[self.fy_mask]
		self.log_gy_in = np.zeros((order, len(self.fy_mask[0])))
		self.log_gy_in[0,:] = self.P[self.fy_mask] - self.Z*self.V[self.fy_mask]
		self.log_dxfy_out = np.zeros((order-1, len(self.fy_mask[0])))
		self.log_dxfy_out[0,:] = self.dxP[self.fy_mask] + self.Z*self.dxV[self.fy_mask]
		self.log_dxgy_in = np.zeros((order, len(self.fy_mask[0])))
		self.log_dxgy_in[0,:] = self.dxP[self.fy_mask] - self.Z*self.dxV[self.fy_mask]
		self.log_dyfy_out = np.zeros((order-1, len(self.fy_mask[0])))
		self.log_dyfy_out[0,:] = self.dyP[self.fy_mask] + self.Z*self.dyV[self.fy_mask]
		self.log_dygy_in = np.zeros((order, len(self.fy_mask[0])))
		self.log_dygy_in[0,:] = self.dyP[self.fy_mask] - self.Z*self.dyV[self.fy_mask]

		self.log_gy_out = np.zeros((order-1, len(self.gy_mask[0])))
		self.log_gy_out[0,:] = self.P[self.gy_mask] - self.Z*self.V[self.gy_mask]
		self.log_fy_in = np.zeros((order, len(self.gy_mask[0])))
		self.log_fy_in[0,:] = self.P[self.gy_mask] + self.Z*self.V[self.gy_mask]
		self.log_dxgy_out = np.zeros((order-1, len(self.gy_mask[0])))
		self.log_dxgy_out[0,:] = self.dxP[self.gy_mask] - self.Z*self.dxV[self.gy_mask]
		self.log_dxfy_in = np.zeros((order, len(self.gy_mask[0])))
		self.log_dxfy_in[0,:] = self.dxP[self.gy_mask] + self.Z*self.dxV[self.gy_mask]
		self.log_dygy_out = np.zeros((order-1, len(self.gy_mask[0])))
		self.log_dygy_out[0,:] = self.dyP[self.gy_mask] - self.Z*self.dyV[self.gy_mask]
		self.log_dyfy_in = np.zeros((order, len(self.gy_mask[0])))
		self.log_dyfy_in[0,:] = self.dyP[self.gy_mask] + self.Z*self.dyV[self.gy_mask]


		self.static = 0

	def boundaryconditionx(self, fi, gi, difi, digi, djfi, djgi):
		self.log_gx_in = np.roll(self.log_gx_in, 1, 0); self.log_gx_in[0,:] = gi[self.fx_mask]
		self.log_dxgx_in = np.roll(self.log_dxgx_in, 1, 0); self.log_dxgx_in[0,:] = digi[self.fx_mask]
		self.log_dygx_in = np.roll(self.log_dygx_in, 1, 0); self.log_dygx_in[0,:] = djgi[self.fx_mask]

		fi[self.fx_mask] = -np.sum(self.log_fx_out*self.Bs_fx, axis = 0) + np.sum(self.log_gx_in*self.As_fx, axis = 0)
		difi[self.fx_mask] = -np.sum(self.log_dxfx_out*self.Bs_fx, axis = 0) - np.sum(self.log_dxgx_in*self.As_fx, axis = 0)
		djfi[self.fx_mask] = -np.sum(self.log_dyfx_out*self.Bs_fx, axis = 0) + np.sum(self.log_dygx_in*self.As_fx, axis = 0)

		self.log_fx_out = np.roll(self.log_fx_out, 1, 0); self.log_fx_out[0,:] = fi[self.fx_mask]
		self.log_dxfx_out = np.roll(self.log_dxfx_out, 1, 0); self.log_dxfx_out[0,:] = difi[self.fx_mask]
		self.log_dyfx_out = np.roll(self.log_dyfx_out, 1, 0); self.log_dyfx_out[0,:] = djfi[self.fx_mask]


		self.log_fx_in = np.roll(self.log_fx_in, 1, 0); self.log_fx_in[0,:] = fi[self.gx_mask]
		self.log_dxfx_in = np.roll(self.log_dxfx_in, 1, 0); self.log_dxfx_in[0,:] = difi[self.gx_mask]
		self.log_dyfx_in = np.roll(self.log_dyfx_in, 1, 0); self.log_dyfx_in[0,:] = djfi[self.gx_mask]

		gi[self.gx_mask] = -np.sum(self.log_gx_out*self.Bs_gx, axis = 0) + np.sum(self.log_fx_in*self.As_gx, axis = 0)
		digi[self.gx_mask] = -np.sum(self.log_dxgx_out*self.Bs_gx, axis = 0) - np.sum(self.log_dxfx_in*self.As_gx, axis = 0)
		djgi[self.gx_mask] = -np.sum(self.log_dygx_out*self.Bs_gx, axis = 0) + np.sum(self.log_dyfx_in*self.As_gx, axis = 0)

		self.log_gx_out = np.roll(self.log_gx_out, 1, 0); self.log_gx_out[0,:] = gi[self.gx_mask]
		self.log_dxgx_out = np.roll(self.log_dxgx_out, 1, 0); self.log_dxgx_out[0,:] = digi[self.gx_mask]
		self.log_dygx_out = np.roll(self.log_dygx_out, 1, 0); self.log_dygx_out[0,:] = djgi[self.gx_mask]

		return fi, gi, difi, digi, djfi, djgi


	def boundaryconditiony(self, fi, gi, difi, digi, djfi, djgi):
		self.log_gy_in = np.roll(self.log_gy_in, 1, 0); self.log_gy_in[0,:] = gi[self.fy_mask]
		self.log_dygy_in = np.roll(self.log_dygy_in, 1, 0); self.log_dygy_in[0,:] = digi[self.fy_mask]
		self.log_dxgy_in = np.roll(self.log_dxgy_in, 1, 0); self.log_dxgy_in[0,:] = djgi[self.fy_mask]

		fi[self.fy_mask] = -np.sum(self.log_fy_out*self.Bs_fy, axis = 0) + np.sum(self.log_gy_in*self.As_fy, axis = 0)
		difi[self.fy_mask] = -np.sum(self.log_dyfy_out*self.Bs_fy, axis = 0) - np.sum(self.log_dygy_in*self.As_fy, axis = 0)
		djfi[self.fy_mask] = -np.sum(self.log_dxfy_out*self.Bs_fy, axis = 0) + np.sum(self.log_dxgy_in*self.As_fy, axis = 0)

		self.log_fy_out = np.roll(self.log_fy_out, 1, 0); self.log_fy_out[0,:] = fi[self.fy_mask]
		self.log_dyfy_out = np.roll(self.log_dyfy_out, 1, 0); self.log_dyfy_out[0,:] = difi[self.fy_mask]
		self.log_dxfy_out = np.roll(self.log_dxfy_out, 1, 0); self.log_dxfy_out[0,:] = djfi[self.fy_mask]


		self.log_fy_in = np.roll(self.log_fy_in, 1, 0); self.log_fy_in[0,:] = fi[self.gy_mask]
		self.log_dyfy_in = np.roll(self.log_dyfy_in, 1, 0); self.log_dyfy_in[0,:] = difi[self.gy_mask]
		self.log_dxfy_in = np.roll(self.log_dxfy_in, 1, 0); self.log_dxfy_in[0,:] = djfi[self.gy_mask]

		gi[self.gy_mask] = -np.sum(self.log_gy_out*self.Bs_gy, axis = 0) + np.sum(self.log_fy_in*self.As_gy, axis = 0)
		digi[self.gy_mask] = -np.sum(self.log_dygy_out*self.Bs_gy, axis = 0) - np.sum(self.log_dyfy_in*self.As_gy, axis = 0)
		djgi[self.gy_mask] = -np.sum(self.log_dxgy_out*self.Bs_gy, axis = 0) + np.sum(self.log_dxfy_in*self.As_gy, axis = 0)

		self.log_gy_out = np.roll(self.log_gy_out, 1, 0); self.log_gy_out[0,:] = gi[self.gy_mask]
		self.log_dygy_out = np.roll(self.log_dygy_out, 1, 0); self.log_dygy_out[0,:] = digi[self.gy_mask]
		self.log_dxgy_out = np.roll(self.log_dxgy_out, 1, 0); self.log_dxgy_out[0,:] = djgi[self.gy_mask]

		return fi, gi, difi, digi, djfi, djgi


"""
/******************************/
M型RCIP法による2次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class MRCIP_IIR(MCIP_IIR):
	def __init__(self, p_init, dxp_init, dyp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5, alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dx, dt, voxel_label, As, Bs, rho, k)
		self.alpha = 1.

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (np.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new

"""
/******************************/
C型CIP法による2次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class CCIP_IIR(MCIP_IIR):
	def __init__(self, p_init, dxp_init, dyp_init, dxyp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5):
		super().__init__(p_init, dxp_init, dyp_init, dx, dt, voxel_label, As, Bs, rho, k)
		order = As.shape[1]
		shape = (p_init.shape[0]+2, p_init.shape[1]+2)
		self.dxyP = np.zeros(shape); self.dxyP[1:-1,1:-1] = dxyp_init; self.dxyP[self.wall_mask] = np.nan
		self.dxyU = np.zeros(shape); self.dxyU[self.wall_mask] = np.nan
		self.dxyV = np.zeros(shape); self.dxyV[self.wall_mask] = np.nan

		self.log_dxyfx_out = np.zeros((order-1, len(self.fx_mask[0])))
		self.log_dxyfx_out[0,:] = self.dxyP[self.fx_mask] + self.Z*self.dxyU[self.fx_mask]
		self.log_dxygx_in = np.zeros((order, len(self.fx_mask[0])))
		self.log_dxygx_in[0,:] = self.dxyP[self.fx_mask] - self.Z*self.dxyU[self.fx_mask]

		self.log_dxygx_out = np.zeros((order-1, len(self.gx_mask[0])))
		self.log_dxygx_out[0,:] = self.dxyP[self.gx_mask] - self.Z*self.dxyU[self.gx_mask]
		self.log_dxyfx_in = np.zeros((order, len(self.gx_mask[0])))
		self.log_dxyfx_in[0,:] = self.dxyP[self.gx_mask] + self.Z*self.dxyU[self.gx_mask]

		self.log_dxyfy_out = np.zeros((order-1, len(self.fy_mask[0])))
		self.log_dxyfy_out[0,:] = self.dxyP[self.fy_mask] + self.Z*self.dxyV[self.fy_mask]
		self.log_dxygy_in = np.zeros((order, len(self.fy_mask[0])))
		self.log_dxygy_in[0,:] = self.dxyP[self.fy_mask] - self.Z*self.dxyV[self.fy_mask]

		self.log_dxygy_out = np.zeros((order-1, len(self.gy_mask[0])))
		self.log_dxygy_out[0,:] = self.dxyP[self.gy_mask] - self.Z*self.dxyV[self.gy_mask]
		self.log_dxyfy_in = np.zeros((order, len(self.gy_mask[0])))
		self.log_dxyfy_in[0,:] = self.dxyP[self.gy_mask] + self.Z*self.dxyV[self.gy_mask]

	def boundaryconditionx(self, fi, gi, difi, digi, djfi, djgi, dijfi, dijgi):
		self.log_gx_in = np.roll(self.log_gx_in, 1, 0); self.log_gx_in[0,:] = gi[self.fx_mask]
		self.log_dxgx_in = np.roll(self.log_dxgx_in, 1, 0); self.log_dxgx_in[0,:] = digi[self.fx_mask]
		self.log_dygx_in = np.roll(self.log_dygx_in, 1, 0); self.log_dygx_in[0,:] = djgi[self.fx_mask]
		self.log_dxygx_in = np.roll(self.log_dxygx_in, 1, 0); self.log_dxygx_in[0,:] = dijgi[self.fx_mask]

		fi[self.fx_mask] = -np.sum(self.log_fx_out*self.Bs_fx, axis = 0) + np.sum(self.log_gx_in*self.As_fx, axis = 0)
		difi[self.fx_mask] = -np.sum(self.log_dxfx_out*self.Bs_fx, axis = 0) - np.sum(self.log_dxgx_in*self.As_fx, axis = 0)
		djfi[self.fx_mask] = -np.sum(self.log_dyfx_out*self.Bs_fx, axis = 0) + np.sum(self.log_dygx_in*self.As_fx, axis = 0)
		dijfi[self.fx_mask] = -np.sum(self.log_dxyfx_out*self.Bs_fx, axis = 0) - np.sum(self.log_dxygx_in*self.As_fx, axis = 0)

		self.log_fx_out = np.roll(self.log_fx_out, 1, 0); self.log_fx_out[0,:] = fi[self.fx_mask]
		self.log_dxfx_out = np.roll(self.log_dxfx_out, 1, 0); self.log_dxfx_out[0,:] = difi[self.fx_mask]
		self.log_dyfx_out = np.roll(self.log_dyfx_out, 1, 0); self.log_dyfx_out[0,:] = djfi[self.fx_mask]
		self.log_dxyfx_out = np.roll(self.log_dxyfx_out, 1, 0); self.log_dxyfx_out[0,:] = dijfi[self.fx_mask]


		self.log_fx_in = np.roll(self.log_fx_in, 1, 0); self.log_fx_in[0,:] = fi[self.gx_mask]
		self.log_dxfx_in = np.roll(self.log_dxfx_in, 1, 0); self.log_dxfx_in[0,:] = difi[self.gx_mask]
		self.log_dyfx_in = np.roll(self.log_dyfx_in, 1, 0); self.log_dyfx_in[0,:] = djfi[self.gx_mask]
		self.log_dxyfx_in = np.roll(self.log_dxyfx_in, 1, 0); self.log_dxyfx_in[0,:] = dijfi[self.gx_mask]

		gi[self.gx_mask] = -np.sum(self.log_gx_out*self.Bs_gx, axis = 0) + np.sum(self.log_fx_in*self.As_gx, axis = 0)
		digi[self.gx_mask] = -np.sum(self.log_dxgx_out*self.Bs_gx, axis = 0) - np.sum(self.log_dxfx_in*self.As_gx, axis = 0)
		djgi[self.gx_mask] = -np.sum(self.log_dygx_out*self.Bs_gx, axis = 0) + np.sum(self.log_dyfx_in*self.As_gx, axis = 0)
		dijgi[self.gx_mask] = -np.sum(self.log_dxygx_out*self.Bs_gx, axis = 0) - np.sum(self.log_dxyfx_in*self.As_gx, axis = 0)

		self.log_gx_out = np.roll(self.log_gx_out, 1, 0); self.log_gx_out[0,:] = gi[self.gx_mask]
		self.log_dxgx_out = np.roll(self.log_dxgx_out, 1, 0); self.log_dxgx_out[0,:] = digi[self.gx_mask]
		self.log_dygx_out = np.roll(self.log_dygx_out, 1, 0); self.log_dygx_out[0,:] = djgi[self.gx_mask]
		self.log_dxygx_out = np.roll(self.log_dxygx_out, 1, 0); self.log_dxygx_out[0,:] = dijgi[self.gx_mask]

		return fi, gi, difi, digi, djfi, djgi, dijfi, dijgi

	def boundaryconditiony(self, fi, gi, difi, digi, djfi, djgi, dijfi, dijgi):
		self.log_gy_in = np.roll(self.log_gy_in, 1, 0); self.log_gy_in[0,:] = gi[self.fy_mask]
		self.log_dygy_in = np.roll(self.log_dygy_in, 1, 0); self.log_dygy_in[0,:] = digi[self.fy_mask]
		self.log_dxgy_in = np.roll(self.log_dxgy_in, 1, 0); self.log_dxgy_in[0,:] = djgi[self.fy_mask]
		self.log_dxygy_in = np.roll(self.log_dxygy_in, 1, 0); self.log_dxygy_in[0,:] = dijgi[self.fy_mask]

		fi[self.fy_mask] = -np.sum(self.log_fy_out*self.Bs_fy, axis = 0) + np.sum(self.log_gy_in*self.As_fy, axis = 0)
		difi[self.fy_mask] = -np.sum(self.log_dyfy_out*self.Bs_fy, axis = 0) - np.sum(self.log_dygy_in*self.As_fy, axis = 0)
		djfi[self.fy_mask] = -np.sum(self.log_dxfy_out*self.Bs_fy, axis = 0) + np.sum(self.log_dxgy_in*self.As_fy, axis = 0)
		dijfi[self.fy_mask] = -np.sum(self.log_dxyfy_out*self.Bs_fy, axis = 0) - np.sum(self.log_dxygy_in*self.As_fy, axis = 0)

		self.log_fy_out = np.roll(self.log_fy_out, 1, 0); self.log_fy_out[0,:] = fi[self.fy_mask]
		self.log_dyfy_out = np.roll(self.log_dyfy_out, 1, 0); self.log_dyfy_out[0,:] = difi[self.fy_mask]
		self.log_dxfy_out = np.roll(self.log_dxfy_out, 1, 0); self.log_dxfy_out[0,:] = djfi[self.fy_mask]
		self.log_dxyfy_out = np.roll(self.log_dxyfy_out, 1, 0); self.log_dxyfy_out[0,:] = dijfi[self.fy_mask]

		self.log_fy_in = np.roll(self.log_fy_in, 1, 0); self.log_fy_in[0,:] = fi[self.gy_mask]
		self.log_dyfy_in = np.roll(self.log_dyfy_in, 1, 0); self.log_dyfy_in[0,:] = difi[self.gy_mask]
		self.log_dxfy_in = np.roll(self.log_dxfy_in, 1, 0); self.log_dxfy_in[0,:] = djfi[self.gy_mask]
		self.log_dxyfy_in = np.roll(self.log_dxyfy_in, 1, 0); self.log_dxyfy_in[0,:] = dijfi[self.gy_mask]

		gi[self.gy_mask] = -np.sum(self.log_gy_out*self.Bs_gy, axis = 0) + np.sum(self.log_fy_in*self.As_gy, axis = 0)
		digi[self.gy_mask] = -np.sum(self.log_dygy_out*self.Bs_gy, axis = 0) - np.sum(self.log_dyfy_in*self.As_gy, axis = 0)
		djgi[self.gy_mask] = -np.sum(self.log_dxgy_out*self.Bs_gy, axis = 0) + np.sum(self.log_dxfy_in*self.As_gy, axis = 0)
		dijgi[self.gy_mask] = -np.sum(self.log_dxygy_out*self.Bs_gy, axis = 0) - np.sum(self.log_dxyfy_in*self.As_gy, axis = 0)

		self.log_gy_out = np.roll(self.log_gy_out, 1, 0); self.log_gy_out[0,:] = gi[self.gy_mask]
		self.log_dygy_out = np.roll(self.log_dygy_out, 1, 0); self.log_dygy_out[0,:] = digi[self.gy_mask]
		self.log_dxgy_out = np.roll(self.log_dxgy_out, 1, 0); self.log_dxgy_out[0,:] = djgi[self.gy_mask]
		self.log_dxygy_out = np.roll(self.log_dxygy_out, 1, 0); self.log_dxygy_out[0,:] = dijgi[self.gy_mask]

		return fi, gi, difi, digi, djfi, djgi, dijfi, dijgi

	def updatex(self):
		##########x方向の移流
		#####fx, dxfx, dyfx, dxyfxの定義
		fx = self.P + self.Z*self.U
		dxfx = self.dxP + self.Z*self.dxU
		dyfx = self.dyP + self.Z*self.dyU
		dxyfx = self.dxyP + self.Z*self.dxyU 
		#####fx, dxfxの更新
		fx[1:,:], dxfx[1:,:] = self.interpolate(fx[1:,:], dxfx[1:,:], fx[:-1,:], dxfx[:-1,:], -self.dx, -self.epsilon)
		#####dyfx, dxyfxの更新
		dyfx[1:,:], dxyfx[1:,:] = self.interpolate(dyfx[1:,:], dxyfx[1:,:], dyfx[:-1,:], dxyfx[:-1,:], -self.dx, -self.epsilon)

		#####gx, dxgx, dygx, dxygxの定義
		gx = self.P - self.Z*self.U
		dxgx = self.dxP - self.Z*self.dxU
		dygx = self.dyP - self.Z*self.dyU
		dxygx = self.dxyP - self.Z*self.dxyU
		#####gx, dxgxの更新
		gx[:-1,:], dxgx[:-1,:] = self.interpolate(gx[:-1,:], dxgx[:-1,:], gx[1:,:], dxgx[1:,:], self.dx, self.epsilon)
		#####dygxの更新
		dygx[:-1,:], dxygx[:-1,:] = self.interpolate(dygx[:-1,:], dxygx[:-1,:], dygx[1:,:], dxygx[1:,:], self.dx, self.epsilon)

		#####境界条件
		fx, gx, dxfx, dxgx, dyfx, dygx, dxyfx, dxygx = self.boundaryconditionx(fx, gx, dxfx, dxgx, dyfx, dygx, dxyfx, dxygx)

		##########物理量の更新
		self.P = (fx + gx)/2.; self.P[self.wall_mask] = np.nan 
		self.U = (fx - gx)/(2.*self.Z); self.U[self.wall_mask] = np.nan
		self.dxP = (dxfx + dxgx)/2.; self.dxP[self.wall_mask] = np.nan
		self.dxU = (dxfx - dxgx)/(2.*self.Z); self.dxU[self.wall_mask] = np.nan
		self.dyP = (dyfx + dygx)/2.; self.dyP[self.wall_mask] = np.nan
		self.dyU = (dyfx - dygx)/(2.*self.Z); self.dyU[self.wall_mask] = np.nan
		self.dxyP = (dxyfx + dxygx)/2.; self.dxyP[self.wall_mask] = np.nan
		self.dxyU = (dxyfx - dxygx)/(2.*self.Z); self.dxyU[self.wall_mask] = np.nan

	def updatey(self):
		##########y方向の移流
		#####fy, dxfy, dyfy, dxyfyの定義
		fy = self.P + self.Z*self.V
		dxfy = self.dxP + self.Z*self.dxV
		dyfy = self.dyP + self.Z*self.dyV
		dxyfy = self.dxyP + self.Z*self.dxyV
		#####fy, dyfyの更新
		fy[:,1:], dyfy[:,1:] = self.interpolate(fy[:,1:], dyfy[:,1:], fy[:,:-1], dyfy[:,:-1], -self.dx, -self.epsilon)
		#####dxfyの更新
		dxfy[:,1:], dxyfy[:,1:] = self.interpolate(dxfy[:,1:], dxyfy[:,1:], dxfy[:,:-1], dxyfy[:,:-1], -self.dx, -self.epsilon)

		#####gy, dxgy, dygyの定義
		gy = self.P-self.Z*self.V
		dxgy = self.dxP-self.Z*self.dxV
		dygy = self.dyP-self.Z*self.dyV
		dxygy = self.dxyP-self.Z*self.dxyV
		
		#####gy, dygyの更新
		gy[:,:-1], dygy[:,:-1] = self.interpolate(gy[:,:-1], dygy[:,:-1], gy[:,1:], dygy[:,1:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgy[:,:-1], dxygy[:,:-1] = self.interpolate(dxgy[:,:-1], dxygy[:,:-1], dxgy[:,1:], dxygy[:,1:], self.dx, self.epsilon)

		#####境界条件
		fy, gy, dyfy, dygy, dxfy, dxgy, dxyfy, dxygy = self.boundaryconditiony(fy, gy, dyfy, dygy, dxfy, dxgy, dxyfy, dxygy)

		##########物理量の更新
		self.P = (fy + gy)/2.; self.P[self.wall_mask] = np.nan
		self.V = (fy - gy)/(2.*self.Z); self.V[self.wall_mask] = np.nan
		self.dyP = (dyfy + dygy)/2.; self.dyP[self.wall_mask] = np.nan
		self.dyV = (dyfy - dygy)/(2.*self.Z); self.dyV[self.wall_mask] = np.nan
		self.dxP = (dxfy + dxgy)/2.; self.dxP[self.wall_mask] = np.nan
		self.dxV = (dxfy - dxgy)/(2.*self.Z); self.dyV[self.wall_mask] = np.nan
		self.dxyP = (dxyfy + dxygy)/2.; self.dxyP[self.wall_mask] = np.nan
		self.dxyV = (dxyfy - dxygy)/(2.*self.Z); self.dxyV[self.wall_mask] = np.nan


"""
/******************************/
C型RCIP法による2次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class CRCIP_IIR(CCIP_IIR):
	def __init__(self, p_init, dxp_init, dyp_init, dxyp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5, alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dxyp_init, dx, dt, voxel_label, As, Bs, rho, k)
		self.alpha = alpha

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (np.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new