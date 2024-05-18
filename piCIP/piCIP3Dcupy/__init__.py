import cupy as cp
import math
import sys
from piCIP.piCIP3Dcupy import util


"""
/*******************************/
M型CIP法による3次元音伝播シミュレーション
/*******************************/
"""
class MCIP:
	"""
	/*****************************/
	process : コンストラクタ
	/*****************************/
	input:
		p_init, dxp_init, dyp_init -> <cp:float:(X, Y)> 圧力初期分布
		dx, dt -> <float> 刻み幅
		voxel_label -> <cp:int:(X+2, Y+2)> 計算格子材料物性ラベル。
		Rs -> <np:float:(max(voxel_label), )> 各材質(ラベル)の反射率。
	Note:
	---voxel_label---
	空気のラベルはゼロ。
	"""
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = cp.ones(1)):
		shape = (p_init.shape[0]+2, p_init.shape[1]+2, p_init.shape[2]+2)
		
		self.Z = math.sqrt(rho*k)
		self.dx = dx
		self.dt = dt
		self.epsilon = math.sqrt(k/rho)*self.dt

		if voxel_label is None:
			voxel_label = np.ones(shape); voxel_label[1:-1,1:-1,1:-1] = 0
		assert cp.max(voxel_label) == len(Rs), "voxel_label error"

		self.fx_mask, self.gx_mask, self.fy_mask, self.gy_mask, self.fz_mask, self.gz_mask, self.wall_mask, self.r_fx, self.r_gx, self.r_fy, self.r_gy, self.r_fz, self.r_gz = util.Label2Mask(voxel_label, Rs)
		self.P = cp.zeros(shape); self.P[1:-1,1:-1,1:-1] = p_init; self.P[self.wall_mask] = cp.nan
		self.dxP = cp.zeros(shape); self.dxP[1:-1,1:-1,1:-1] = dxp_init; self.dxP[self.wall_mask] = cp.nan
		self.dyP = cp.zeros(shape); self.dyP[1:-1,1:-1,1:-1] = dyp_init; self.dyP[self.wall_mask] = cp.nan
		self.dzP = cp.zeros(shape); self.dzP[1:-1,1:-1,1:-1] = dzp_init; self.dzP[self.wall_mask] = cp.nan

		self.U = cp.zeros(shape); self.U[self.wall_mask] = cp.nan
		self.V = cp.zeros(shape); self.V[self.wall_mask] = cp.nan
		self.W = cp.zeros(shape); self.W[self.wall_mask] = cp.nan
		self.dxU = cp.zeros(shape); self.dxU[self.wall_mask] = cp.nan
		self.dyU = cp.zeros(shape); self.dyU[self.wall_mask] = cp.nan
		self.dzU = cp.zeros(shape); self.dzU[self.wall_mask] = cp.nan
		self.dxV = cp.zeros(shape); self.dxV[self.wall_mask] = cp.nan
		self.dyV = cp.zeros(shape); self.dyV[self.wall_mask] = cp.nan
		self.dzV = cp.zeros(shape); self.dzV[self.wall_mask] = cp.nan
		self.dxW = cp.zeros(shape); self.dxW[self.wall_mask] = cp.nan
		self.dyW = cp.zeros(shape); self.dyW[self.wall_mask] = cp.nan
		self.dzW = cp.zeros(shape); self.dzW[self.wall_mask] = cp.nan

		self.static = 0

	def getP(self, ghost_val = None):
		P = cp.copy(self.P)
		if not(ghost_val is None):
			P[self.wall_mask] = ghost_val
		return P[1:-1,1:-1,1:-1]

	def getVel(self, ghost_val = None):
		U = cp.copy(self.U); V = cp.copy(self.V); W = cp.copy(self.W)
		if not(ghost_val is None):
			U[self.wall_mask] = ghost_val; V[self.wall_mask] = ghost_val; W[self.wall_mask] = ghost_val
		Vel = np.stack((U[1:-1,1:-1,1:-1], V[1:-1,1:-1,1:-1], W[1:-1,1:-1,1:-1]), axis = -1)
		return Vel

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		a = (df + dfup)/(D**2) + 2.*(f-fup)/(D**3)
		b = 3.*(fup-f)/(D**2) - (2.*df+dfup)/D

		f_new = a*(epsilon**3) + b*(epsilon**2) + df*epsilon + f
		df_new = 3.*a*(epsilon**2) + 2.*b*epsilon + df

		return f_new, df_new

	def boundaryconditionx(self, fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx):
		fx[self.fx_mask] = gx[self.fx_mask] * self.r_fx
		dxfx[self.fx_mask] = -dxgx[self.fx_mask] * self.r_fx
		dyfx[self.fx_mask] = dygx[self.fx_mask] * self.r_fx
		dzfx[self.fx_mask] = dzgx[self.fx_mask] * self.r_fx

		gx[self.gx_mask] = fx[self.gx_mask] * self.r_gx
		dxgx[self.gx_mask] = -dxfx[self.gx_mask] * self.r_gx
		dygx[self.gx_mask] = dyfx[self.gx_mask] * self.r_gx
		dzgx[self.gx_mask] = dzfx[self.gx_mask] * self.r_gx

		return fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx

	def boundaryconditiony(self, fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy):
		fy[self.fy_mask] = gy[self.fy_mask] * self.r_fy
		dyfy[self.fy_mask] = -dygy[self.fy_mask] * self.r_fy
		dxfy[self.fy_mask] = dxgy[self.fy_mask] * self.r_fy
		dzfy[self.fy_mask] = dzgy[self.fy_mask] * self.r_fy

		gy[self.gy_mask] = fy[self.gy_mask] * self.r_gy
		dygy[self.gy_mask] = -dyfy[self.gy_mask] * self.r_gy
		dxgy[self.gy_mask] = dxfy[self.gy_mask] * self.r_gy
		dzgy[self.gy_mask] = dzfy[self.gy_mask] * self.r_gy

		return fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy

	def boundaryconditionz(self, fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz):
		fz[self.fz_mask] = gz[self.fz_mask] * self.r_fz
		dzfz[self.fz_mask] = -dzgz[self.fz_mask] * self.r_fz
		dxfz[self.fz_mask] = dxgz[self.fz_mask] * self.r_fz
		dyfz[self.fz_mask] = dygz[self.fz_mask] * self.r_fz

		gz[self.gz_mask] = fz[self.gz_mask] * self.r_gz
		dzgz[self.gz_mask] = -dzfz[self.gz_mask] * self.r_gz
		dxgz[self.gz_mask] = dxfz[self.gz_mask] * self.r_gz
		dygz[self.gz_mask] = dyfz[self.gz_mask] * self.r_gz

		return fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz

	def updatex(self):
		##########x方向の移流
		#####fx, dxfx, dyfx, dzfxの定義
		fx = self.P + self.Z*self.U
		dxfx = self.dxP+self.Z*self.dxU
		dyfx = self.dyP+self.Z*self.dyU
		dzfx = self.dzP+self.Z*self.dzU
		#####fx, dxfxの更新
		fx[1:,:,:], dxfx[1:,:,:] = self.interpolate(fx[1:,:,:], dxfx[1:,:,:], fx[:-1,:,:], dxfx[:-1,:,:], -self.dx, -self.epsilon)

		#####dyfx, dzfxの更新
		dyfx[1:,:,:] = (1.-self.epsilon/self.dx)*dyfx[1:,:,:] + (self.epsilon/self.dx)*dyfx[:-1,:,:]
		dzfx[1:,:,:] = (1.-self.epsilon/self.dx)*dzfx[1:,:,:] + (self.epsilon/self.dx)*dzfx[:-1,:,:]

		#####gx, dxgx, dygx, dzgxの定義
		gx = self.P-self.Z*self.U
		dxgx = self.dxP-self.Z*self.dxU
		dygx = self.dyP-self.Z*self.dyU
		dzgx = self.dzP-self.Z*self.dzU
		#####gx, dxgxの更新
		gx[:-1,:,:], dxgx[:-1,:,:] = self.interpolate(gx[:-1,:,:], dxgx[:-1,:,:], gx[1:,:,:], dxgx[1:,:,:], self.dx, self.epsilon)

		#####dygx, dzgxの更新
		dygx[:-1,:,:] = (1. - self.epsilon/self.dx)*dygx[:-1,:,:] + (self.epsilon/self.dx)*dygx[1:,:,:]
		dzgx[:-1,:,:] = (1. - self.epsilon/self.dx)*dzgx[:-1,:,:] + (self.epsilon/self.dx)*dzgx[1:,:,:]

		#####境界条件
		fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx = self.boundaryconditionx(fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx)

		##########物理量の更新
		self.P = (fx + gx)/2.; self.P[self.wall_mask] = cp.nan
		self.U = (fx - gx)/(2.*self.Z); self.U[self.wall_mask] = cp.nan 
		self.dxP = (dxfx + dxgx)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxU = (dxfx - dxgx)/(2.*self.Z); self.dxU[self.wall_mask] = cp.nan
		self.dyP = (dyfx + dygx)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyU = (dyfx - dygx)/(2.*self.Z); self.dyU[self.wall_mask] = cp.nan
		self.dzP = (dzfx + dzgx)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzU = (dzfx - dzgx)/(2.*self.Z); self.dzU[self.wall_mask] = cp.nan

	def updatey(self):
		##########y方向の移流
		#####fy, dxfy, dyfy, dzfyの定義
		fy = self.P + self.Z*self.V
		dxfy = self.dxP + self.Z*self.dxV
		dyfy = self.dyP + self.Z*self.dyV
		dzfy = self.dzP + self.Z*self.dzV
		#####fy, dyfyの更新
		fy[:,1:,:], dyfy[:,1:,:] = self.interpolate(fy[:,1:,:], dyfy[:,1:,:], fy[:,:-1,:], dyfy[:,:-1,:], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfy[:,1:,:] = (1.-self.epsilon/self.dx)*dxfy[:,1:,:] + (self.epsilon/self.dx)*dxfy[:,:-1,:]
		dzfy[:,1:,:] = (1.-self.epsilon/self.dx)*dzfy[:,1:,:] + (self.epsilon/self.dx)*dzfy[:,:-1,:]

		#####gy, dxgy, dygyの定義
		gy = self.P - self.Z*self.V
		dxgy = self.dxP - self.Z*self.dxV
		dygy = self.dyP - self.Z*self.dyV
		dzgy = self.dzP - self.Z*self.dzV
		#####gy, dygyの更新
		gy[:,:-1,:], dygy[:,:-1,:] = self.interpolate(gy[:,:-1,:], dygy[:,:-1,:], gy[:,1:,:], dygy[:,1:,:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgy[:,:-1,:] = (1.-self.epsilon/self.dx)*dxgy[:,:-1,:] + (self.epsilon/self.dx)*dxgy[:,1:,:]
		dzgy[:,:-1,:] = (1.-self.epsilon/self.dx)*dzgy[:,:-1,:] + (self.epsilon/self.dx)*dzgy[:,1:,:]

		#####境界条件
		fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy = self.boundaryconditiony(fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy)
		
		##########物理量の更新
		self.P = (fy + gy)/2.; self.P[self.wall_mask] = cp.nan
		self.V = (fy - gy)/(2.*self.Z); self.V[self.wall_mask] = cp.nan
		self.dyP = (dyfy + dygy)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyV = (dyfy - dygy)/(2.*self.Z); self.dyV[self.wall_mask] = cp.nan
		self.dxP = (dxfy + dxgy)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxV = (dxfy - dxgy)/(2.*self.Z); self.dxV[self.wall_mask] = cp.nan
		self.dzP = (dzfy + dzgy)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzV = (dzfy - dzgy)/(2.*self.Z); self.dzV[self.wall_mask] = cp.nan

	def updatez(self):
		##########z方向の移流
		#####fz, dxfz, dyfz, dzfzの定義
		fz = self.P + self.Z*self.W
		dxfz = self.dxP + self.Z*self.dxW
		dyfz = self.dyP + self.Z*self.dyW
		dzfz = self.dzP + self.Z*self.dzW
		#####fy, dyfyの更新
		fz[:,:,1:], dzfz[:,:,1:] = self.interpolate(fz[:,:,1:], dzfz[:,:,1:], fz[:,:,:-1], dzfz[:,:,:-1], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfz[:,:,1:] = (1.-self.epsilon/self.dx)*dxfz[:,:,1:] + (self.epsilon/self.dx)*dxfz[:,:,:-1]
		dyfz[:,:,1:] = (1.-self.epsilon/self.dx)*dyfz[:,:,1:] + (self.epsilon/self.dx)*dyfz[:,:,:-1]

		#####gy, dxgy, dygyの定義
		gz = self.P - self.Z*self.W
		dxgz = self.dxP - self.Z*self.dxW
		dygz = self.dyP - self.Z*self.dyW
		dzgz = self.dzP - self.Z*self.dzW
		#####gy, dygyの更新
		gz[:,:,:-1], dzgz[:,:,:-1] = self.interpolate(gz[:,:,:-1], dzgz[:,:,:-1], gz[:,:,1:], dzgz[:,:,1:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgz[:,:,:-1] = (1.-self.epsilon/self.dx)*dxgz[:,:,:-1] + (self.epsilon/self.dx)*dxgz[:,:,1:]
		dygz[:,:,:-1] = (1.-self.epsilon/self.dx)*dygz[:,:,:-1] + (self.epsilon/self.dx)*dygz[:,:,1:]

		#####境界条件
		fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz = self.boundaryconditionz(fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz)	

		##########物理量の更新
		self.P = (fz + gz)/2.; self.P[self.wall_mask] = cp.nan
		self.W = (fz - gz)/(2.*self.Z); self.W[self.wall_mask] = cp.nan
		self.dyP = (dyfz + dygz)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyW = (dyfz - dygz)/(2.*self.Z); self.dyW[self.wall_mask] = cp.nan
		self.dxP = (dxfz + dxgz)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxW = (dxfz - dxgz)/(2.*self.Z); self.dxW[self.wall_mask] = cp.nan
		self.dzP = (dzfz + dzgz)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzW = (dzfz - dzgz)/(2.*self.Z); self.dzW[self.wall_mask] = cp.nan

	def update(self):
		if (self.static%3) == 0:
			self.updatex(); self.updatey(); self.updatez()
		elif (self.static%3) == 1:
			self.updatey(); self.updatez(); self.updatex()
		else:
			self.updatez(); self.updatex(); self.updatey()

		self.static += 1

"""
/*******************************/
M型RCIP法による3次元音伝播シミュレーション
/*******************************/
"""
class MRCIP(MCIP):
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = cp.ones(1), alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dzp_init, dx, dt, rho, k, voxel_label, Rs)
		self.alpha = alpha

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (cp.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new

"""
/*******************************/
C型CIP法による3次元音伝播シミュレーション
/*******************************/
"""
class CCIP(MCIP):
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dxyp_init, dyzp_init, dxzp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = cp.ones(1)):
		super().__init__(p_init, dxp_init, dyp_init, dzp_init, dx, dt, rho, k, voxel_label, Rs)
		shape = (p_init.shape[0]+2, p_init.shape[1]+2, p_init.shape[2]+2)
		self.dxyP = cp.zeros(shape); self.dxyP[1:-1,1:-1,1:-1] = dxyp_init; self.dxyP[self.wall_mask] = cp.nan
		self.dxyU = cp.zeros(shape); self.dxyU[self.wall_mask] = cp.nan
		self.dxyV = cp.zeros(shape); self.dxyV[self.wall_mask] = cp.nan
		self.dxyW = cp.zeros(shape); self.dxyW[self.wall_mask] = cp.nan

		self.dyzP = cp.zeros(shape); self.dyzP[1:-1,1:-1,1:-1] = dyzp_init; self.dyzP[self.wall_mask] = cp.nan
		self.dyzU = cp.zeros(shape); self.dyzU[self.wall_mask] = cp.nan
		self.dyzV = cp.zeros(shape); self.dyzV[self.wall_mask] = cp.nan
		self.dyzW = cp.zeros(shape); self.dyzW[self.wall_mask] = cp.nan

		self.dxzP = cp.zeros(shape); self.dxzP[1:-1,1:-1,1:-1] = dxzp_init; self.dxzP[self.wall_mask] = cp.nan
		self.dxzU = cp.zeros(shape); self.dxzU[self.wall_mask] = cp.nan
		self.dxzV = cp.zeros(shape); self.dxzV[self.wall_mask] = cp.nan
		self.dxzW = cp.zeros(shape); self.dxzW[self.wall_mask] = cp.nan

	def boundaryconditionx(self, fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx):
		fx[self.fx_mask] = gx[self.fx_mask] * self.r_fx
		dxfx[self.fx_mask] = -dxgx[self.fx_mask] * self.r_fx
		dyfx[self.fx_mask] = dygx[self.fx_mask] * self.r_fx
		dzfx[self.fx_mask] = dzgx[self.fx_mask] * self.r_fx
		dxyfx[self.fx_mask] = -dxygx[self.fx_mask] * self.r_fx
		dxzfx[self.fx_mask] = -dxzgx[self.fx_mask] * self.r_fx

		gx[self.gx_mask] = fx[self.gx_mask] * self.r_gx
		dxgx[self.gx_mask] = -dxfx[self.gx_mask] * self.r_gx
		dygx[self.gx_mask] = dyfx[self.gx_mask] * self.r_gx
		dzgx[self.gx_mask] = dzfx[self.gx_mask] * self.r_gx
		dxygx[self.gx_mask] = -dxyfx[self.gx_mask] * self.r_gx
		dxzgx[self.gx_mask] = -dxzfx[self.gx_mask] * self.r_gx

		return fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx

	def boundaryconditiony(self, fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy):
		fy[self.fy_mask] = gy[self.fy_mask] * self.r_fy
		dyfy[self.fy_mask] = -dygy[self.fy_mask] * self.r_fy
		dxfy[self.fy_mask] = dxgy[self.fy_mask] * self.r_fy
		dzfy[self.fy_mask] = dzgy[self.fy_mask] * self.r_fy
		dxyfy[self.fy_mask] = -dxygy[self.fy_mask] * self.r_fy
		dyzfy[self.fy_mask] = -dyzgy[self.fy_mask] * self.r_fy

		gy[self.gy_mask] = fy[self.gy_mask] * self.r_gy
		dygy[self.gy_mask] = -dyfy[self.gy_mask] * self.r_gy
		dxgy[self.gy_mask] = dxfy[self.gy_mask] * self.r_gy
		dzgy[self.gy_mask] = dzfy[self.gy_mask] * self.r_gy
		dxygy[self.gy_mask] = -dxyfy[self.gy_mask] * self.r_gy
		dyzgy[self.gy_mask] = -dyzfy[self.gy_mask] * self.r_gy

		return fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy

	def boundaryconditionz(self, fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz):
		fz[self.fz_mask] = gz[self.fz_mask] * self.r_fz
		dzfz[self.fz_mask] = -dzgz[self.fz_mask] * self.r_fz
		dxfz[self.fz_mask] = dxgz[self.fz_mask] * self.r_fz
		dyfz[self.fz_mask] = dygz[self.fz_mask] * self.r_fz
		dxzfz[self.fz_mask] = -dxzgz[self.fz_mask] * self.r_fz
		dyzfz[self.fz_mask] = -dyzgz[self.fz_mask] * self.r_fz

		gz[self.gz_mask] = fz[self.gz_mask] * self.r_gz
		dzgz[self.gz_mask] = -dzfz[self.gz_mask] * self.r_gz
		dxgz[self.gz_mask] = dxfz[self.gz_mask] * self.r_gz
		dygz[self.gz_mask] = dyfz[self.gz_mask] * self.r_gz
		dxzgz[self.gz_mask] = -dxzfz[self.gz_mask] * self.r_gz
		dyzgz[self.gz_mask] = -dyzfz[self.gz_mask] * self.r_gz

		return fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz

	def updatex(self):
		##########x方向の移流
		#####fx, dxfx, dyfx, dzfxの定義
		fx = self.P + self.Z*self.U
		dxfx = self.dxP + self.Z*self.dxU
		dyfx = self.dyP + self.Z*self.dyU
		dzfx = self.dzP + self.Z*self.dzU
		dxyfx = self.dxyP + self.Z*self.dxyU
		dxzfx = self.dxzP + self.Z*self.dxzU

		#####fx, dxfxの更新
		fx[1:,:,:], dxfx[1:,:,:] = self.interpolate(fx[1:,:,:], dxfx[1:,:,:], fx[:-1,:,:], dxfx[:-1,:,:], -self.dx, -self.epsilon)

		#####dyfx, dzfxの更新
		dyfx[1:,:,:], dxyfx[1:,:,:] = self.interpolate(dyfx[1:,:,:], dxyfx[1:,:,:], dyfx[:-1,:,:], dxyfx[:-1,:,:], -self.dx, -self.epsilon)
		dzfx[1:,:,:], dxzfx[1:,:,:] = self.interpolate(dzfx[1:,:,:], dxzfx[1:,:,:], dzfx[:-1,:,:], dxzfx[:-1,:,:], -self.dx, -self.epsilon)

		#####gx, dxgx, dygx, dzgxの定義
		gx = self.P - self.Z*self.U
		dxgx = self.dxP - self.Z*self.dxU
		dygx = self.dyP - self.Z*self.dyU
		dzgx = self.dzP - self.Z*self.dzU
		dxygx = self.dxyP - self.Z*self.dxyU
		dxzgx = self.dxzP - self.Z*self.dxzU

		#####gx, dxgxの更新
		gx[:-1,:,:], dxgx[:-1,:,:] = self.interpolate(gx[:-1,:,:], dxgx[:-1,:,:], gx[1:,:,:], dxgx[1:,:,:], self.dx, self.epsilon)

		#####dygxの更新
		dygx[:-1,:,:], dxygx[:-1,:,:] = self.interpolate(dygx[:-1,:,:], dxygx[:-1,:,:], dygx[1:,:,:], dxygx[1:,:,:], self.dx, self.epsilon)
		dzgx[:-1,:,:], dxzgx[:-1,:,:] = self.interpolate(dzgx[:-1,:,:], dxzgx[:-1,:,:], dzgx[1:,:,:], dxzgx[1:,:,:], self.dx, self.epsilon)

		#####境界条件
		fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx = self.boundaryconditionx(fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx)
		

		##########物理量の更新
		self.P = (fx + gx)/2.; self.P[self.wall_mask] = cp.nan
		self.U = (fx - gx)/(2.*self.Z); self.U[self.wall_mask] = cp.nan 
		self.dxP = (dxfx + dxgx)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxU = (dxfx - dxgx)/(2.*self.Z); self.dxU[self.wall_mask] = cp.nan
		self.dyP = (dyfx + dygx)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyU = (dyfx - dygx)/(2.*self.Z); self.dyU[self.wall_mask] = cp.nan
		self.dzP = (dzfx + dzgx)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzU = (dzfx - dzgx)/(2.*self.Z); self.dzU[self.wall_mask] = cp.nan
		self.dxyP = (dxyfx + dxygx)/2.; self.dxyP[self.wall_mask] = cp.nan
		self.dxyU = (dxyfx - dxygx)/(2.*self.Z); self.dxyU[self.wall_mask] = cp.nan
		self.dxzP = (dxzfx + dxzgx)/2.; self.dxzP[self.wall_mask] = cp.nan
		self.dxzU = (dxzfx - dxzgx)/(2.*self.Z); self.dxzU[self.wall_mask] = cp.nan



	def updatey(self):
		##########y方向の移流
		#####fy, dxfy, dyfy, dzfyの定義
		fy = self.P + self.Z*self.V
		dxfy = self.dxP + self.Z*self.dxV
		dyfy = self.dyP + self.Z*self.dyV
		dzfy = self.dzP + self.Z*self.dzV
		dxyfy = self.dxyP + self.Z*self.dxyV
		dyzfy = self.dyzP + self.Z*self.dyzV
		#####fy, dyfyの更新
		fy[:,1:,:], dyfy[:,1:,:] = self.interpolate(fy[:,1:,:], dyfy[:,1:,:], fy[:,:-1,:], dyfy[:,:-1,:], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfy[:,1:,:], dxyfy[:,1:,:] = self.interpolate(dxfy[:,1:,:], dxyfy[:,1:,:], dxfy[:,:-1,:], dxyfy[:,:-1,:], -self.dx, -self.epsilon)
		dzfy[:,1:,:], dyzfy[:,1:,:] = self.interpolate(dzfy[:,1:,:], dyzfy[:,1:,:], dzfy[:,:-1,:], dyzfy[:,:-1,:], -self.dx, -self.epsilon)

		#####gy, dxgy, dygyの定義
		gy = self.P - self.Z*self.V
		dxgy = self.dxP - self.Z*self.dxV
		dygy = self.dyP - self.Z*self.dyV
		dzgy = self.dzP - self.Z*self.dzV
		dxygy = self.dxyP - self.Z*self.dxyV
		dyzgy = self.dyzP - self.Z*self.dyzV
		#####gy, dygyの更新
		gy[:,:-1,:], dygy[:,:-1,:] = self.interpolate(gy[:,:-1,:], dygy[:,:-1,:], gy[:,1:,:], dygy[:,1:,:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgy[:,:-1,:], dxygy[:,:-1,:] = self.interpolate(dxgy[:,:-1,:], dxygy[:,:-1,:], dxgy[:,1:,:], dxygy[:,1:,:], self.dx, self.epsilon)
		dzgy[:,:-1,:], dyzgy[:,:-1,:] = self.interpolate(dzgy[:,:-1,:], dyzgy[:,:-1,:], dzgy[:,1:,:], dyzgy[:,1:,:], self.dx, self.epsilon)

		#####境界条件
		fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy = self.boundaryconditiony(fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy)

		##########物理量の更新
		self.P = (fy + gy)/2.; self.P[self.wall_mask] = cp.nan
		self.V = (fy - gy)/(2.*self.Z); self.V[self.wall_mask] = cp.nan 
		self.dxP = (dxfy + dxgy)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxV = (dxfy - dxgy)/(2.*self.Z); self.dxV[self.wall_mask] = cp.nan
		self.dyP = (dyfy + dygy)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyV = (dyfy - dygy)/(2.*self.Z); self.dyV[self.wall_mask] = cp.nan
		self.dzP = (dzfy + dzgy)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzV = (dzfy - dzgy)/(2.*self.Z); self.dzV[self.wall_mask] = cp.nan
		self.dxyP = (dxyfy + dxygy)/2.; self.dxyP[self.wall_mask] = cp.nan
		self.dxyV = (dxyfy - dxygy)/(2.*self.Z); self.dxyV[self.wall_mask] = cp.nan
		self.dyzP = (dyzfy + dyzgy)/2.; self.dyzP[self.wall_mask] = cp.nan
		self.dyzV = (dyzfy - dyzgy)/(2.*self.Z); self.dyzV[self.wall_mask] = cp.nan

	def updatez(self):
		##########z方向の移流
		#####fz, dxfz, dyfz, dzfzの定義
		fz = self.P + self.Z*self.W
		dxfz = self.dxP + self.Z*self.dxW
		dyfz = self.dyP + self.Z*self.dyW
		dzfz = self.dzP + self.Z*self.dzW
		dxzfz = self.dxzP + self.Z*self.dxzW
		dyzfz = self.dyzP + self.Z*self.dyzW
		#####fy, dyfyの更新
		fz[:,:,1:], dzfz[:,:,1:] = self.interpolate(fz[:,:,1:], dzfz[:,:,1:], fz[:,:,:-1], dzfz[:,:,:-1], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfz[:,:,1:], dxzfz[:,:,1:] = self.interpolate(dxfz[:,:,1:], dxzfz[:,:,1:], dxfz[:,:,:-1], dxzfz[:,:,:-1], -self.dx, -self.epsilon)
		dyfz[:,:,1:], dyzfz[:,:,1:] = self.interpolate(dyfz[:,:,1:], dyzfz[:,:,1:], dyfz[:,:,:-1], dyzfz[:,:,:-1], -self.dx, -self.epsilon)

		#####gy, dxgy, dygyの定義
		gz = self.P - self.Z*self.W
		dxgz = self.dxP - self.Z*self.dxW
		dygz = self.dyP - self.Z*self.dyW
		dzgz = self.dzP - self.Z*self.dzW
		dxzgz = self.dxzP - self.Z*self.dxzW
		dyzgz = self.dyzP - self.Z*self.dyzW
		#####gy, dygyの更新
		gz[:,:,:-1], dzgz[:,:,:-1] = self.interpolate(gz[:,:,:-1], dzgz[:,:,:-1], gz[:,:,1:], dzgz[:,:,1:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgz[:,:,:-1], dxzgz[:,:,:-1] = self.interpolate(dxgz[:,:,:-1], dxzgz[:,:,:-1], dxgz[:,:,1:], dxzgz[:,:,1:], self.dx, self.epsilon)
		dygz[:,:,:-1], dyzgz[:,:,:-1] = self.interpolate(dygz[:,:,:-1], dyzgz[:,:,:-1], dygz[:,:,1:], dyzgz[:,:,1:], self.dx, self.epsilon)

		#####境界条件
		fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz = self.boundaryconditionz(fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz)
		
		##########物理量の更新
		self.P = (fz + gz)/2.; self.P[self.wall_mask] = cp.nan
		self.W = (fz - gz)/(2.*self.Z); self.W[self.wall_mask] = cp.nan 
		self.dxP = (dxfz + dxgz)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxW = (dxfz - dxgz)/(2.*self.Z); self.dxW[self.wall_mask] = cp.nan
		self.dyP = (dyfz + dygz)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyW = (dyfz - dygz)/(2.*self.Z); self.dyW[self.wall_mask] = cp.nan
		self.dzP = (dzfz + dzgz)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzW = (dzfz - dzgz)/(2.*self.Z); self.dzW[self.wall_mask] = cp.nan
		self.dxzP = (dxzfz + dxzgz)/2.; self.dxzP[self.wall_mask] = cp.nan
		self.dxzW = (dxzfz - dxzgz)/(2.*self.Z); self.dxzW[self.wall_mask] = cp.nan
		self.dyzP = (dyzfz + dyzgz)/2.; self.dyzP[self.wall_mask] = cp.nan
		self.dyzW = (dyzfz - dyzgz)/(2.*self.Z); self.dyzW[self.wall_mask] = cp.nan


"""
/************************************/
C型RCIP法による3次元音伝播シミュレーション
/************************************/
"""
class CRCIP(CCIP):
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dxyp_init, dyzp_init, dxzp_init, dx, dt, rho = 1.293, k = 1.4e+5, voxel_label = None, Rs = cp.ones(1), alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dzp_init, dxyp_init, dyzp_init, dxzp_init, dx, dt, rho, k, voxel_label, Rs)
		self.alpha = 1.

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (cp.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new

"""
/******************************/
M型CIP法による3次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class MCIP_IIR(MCIP):
	"""
	/*********************/
	process : コンストラクタ
	/*********************/
	input:
		As -> <cp:float:(max(voxel_label), order)> IIRフィルタ係数
		Bs -> <cp:float:(max(voxel_label), order-1)> IIRフィルタ係数
	"""
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5):
		assert cp.max(voxel_label) == len(As), "voxel_label error"

		order = As.shape[1]
		shape = (p_init.shape[0]+2, p_init.shape[1]+2, p_init.shape[2]+2)

		self.Z = math.sqrt(rho*k)
		self.dx = dx
		self.dt = dt
		self.epsilon = math.sqrt(k/rho)*self.dt

		self.fx_mask, self.gx_mask, self.fy_mask, self.gy_mask, self.fz_mask, self.gz_mask, self.wall_mask, self.As_fx, self.Bs_fx, self.As_gx, self.Bs_gx, self.As_fy, self.Bs_fy, self.As_gy, self.Bs_gy, self.As_fz, self.Bs_fz, self.As_gz, self.Bs_gz = util.Label2Mask_IIR(voxel_label, As, Bs)

		self.P = cp.zeros(shape); self.P[1:-1,1:-1,1:-1] = p_init; self.P[self.wall_mask] = cp.nan
		self.dxP = cp.zeros(shape); self.dxP[1:-1,1:-1,1:-1] = dxp_init; self.dxP[self.wall_mask] = cp.nan
		self.dyP = cp.zeros(shape); self.dyP[1:-1,1:-1,1:-1] = dyp_init; self.dyP[self.wall_mask] = cp.nan
		self.dzP = cp.zeros(shape); self.dzP[1:-1,1:-1,1:-1] = dzp_init; self.dzP[self.wall_mask] = cp.nan

		self.U = cp.zeros(shape); self.U[self.wall_mask] = cp.nan
		self.V = cp.zeros(shape); self.V[self.wall_mask] = cp.nan
		self.W = cp.zeros(shape); self.W[self.wall_mask] = cp.nan
		self.dxU = cp.zeros(shape); self.dxU[self.wall_mask] = cp.nan
		self.dyU = cp.zeros(shape); self.dyU[self.wall_mask] = cp.nan
		self.dzU = cp.zeros(shape); self.dzU[self.wall_mask] = cp.nan
		self.dxV = cp.zeros(shape); self.dxV[self.wall_mask] = cp.nan
		self.dyV = cp.zeros(shape); self.dyV[self.wall_mask] = cp.nan
		self.dzV = cp.zeros(shape); self.dzV[self.wall_mask] = cp.nan
		self.dxW = cp.zeros(shape); self.dxW[self.wall_mask] = cp.nan
		self.dyW = cp.zeros(shape); self.dyW[self.wall_mask] = cp.nan
		self.dzW = cp.zeros(shape); self.dzW[self.wall_mask] = cp.nan


		self.log_fx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_fx_out[0,:] = self.P[self.fx_mask] + self.Z*self.U[self.fx_mask]
		self.log_gx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_gx_in[0,:] = self.P[self.fx_mask] - self.Z*self.U[self.fx_mask]
		self.log_dxfx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_dxfx_out[0,:] = self.dxP[self.fx_mask] + self.Z*self.dxU[self.fx_mask]
		self.log_dxgx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_dxgx_in[0,:] = self.dxP[self.fx_mask] - self.Z*self.dxU[self.fx_mask]
		self.log_dyfx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_dyfx_out[0,:] = self.dyP[self.fx_mask] + self.Z*self.dyU[self.fx_mask]
		self.log_dygx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_dygx_in[0,:] = self.dyP[self.fx_mask] - self.Z*self.dyU[self.fx_mask]
		self.log_dzfx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_dzfx_out[0,:] = self.dzP[self.fx_mask] + self.Z*self.dzU[self.fx_mask]
		self.log_dzgx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_dzgx_in[0,:] = self.dzP[self.fx_mask] - self.Z*self.dzU[self.fx_mask]

		self.log_gx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_gx_out[0,:] = self.P[self.gx_mask] - self.Z*self.U[self.gx_mask]
		self.log_fx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_fx_in[0,:] = self.P[self.gx_mask] + self.Z*self.U[self.gx_mask]
		self.log_dxgx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_dxgx_out[0,:] = self.dxP[self.gx_mask] - self.Z*self.dxU[self.gx_mask]
		self.log_dxfx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_dxfx_in[0,:] = self.dxP[self.gx_mask] + self.Z*self.dxU[self.gx_mask]
		self.log_dygx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_dygx_out[0,:] = self.dyP[self.gx_mask] - self.Z*self.dyU[self.gx_mask]
		self.log_dyfx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_dyfx_in[0,:] = self.dyP[self.gx_mask] + self.Z*self.dyU[self.gx_mask]
		self.log_dzgx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_dzgx_out[0,:] = self.dzP[self.gx_mask] - self.Z*self.dzU[self.gx_mask]
		self.log_dzfx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_dzfx_in[0,:] = self.dzP[self.gx_mask] + self.Z*self.dzU[self.gx_mask]

		self.log_fy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_fy_out[0,:] = self.P[self.fy_mask] + self.Z*self.V[self.fy_mask]
		self.log_gy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_gy_in[0,:] = self.P[self.fy_mask] - self.Z*self.V[self.fy_mask]
		self.log_dxfy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_dxfy_out[0,:] = self.dxP[self.fy_mask] + self.Z*self.dxV[self.fy_mask]
		self.log_dxgy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_dxgy_in[0,:] = self.dxP[self.fy_mask] - self.Z*self.dxV[self.fy_mask]
		self.log_dyfy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_dyfy_out[0,:] = self.dyP[self.fy_mask] + self.Z*self.dyV[self.fy_mask]
		self.log_dygy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_dygy_in[0,:] = self.dyP[self.fy_mask] - self.Z*self.dyV[self.fy_mask]
		self.log_dzfy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_dzfy_out[0,:] = self.dzP[self.fy_mask] + self.Z*self.dzV[self.fy_mask]
		self.log_dzgy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_dzgy_in[0,:] = self.dzP[self.fy_mask] - self.Z*self.dzV[self.fy_mask]

		self.log_gy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_gy_out[0,:] = self.P[self.gy_mask] - self.Z*self.V[self.gy_mask]
		self.log_fy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_fy_in[0,:] = self.P[self.gy_mask] + self.Z*self.V[self.gy_mask]
		self.log_dxgy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_dxgy_out[0,:] = self.dxP[self.gy_mask] - self.Z*self.dxV[self.gy_mask]
		self.log_dxfy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_dxfy_in[0,:] = self.dxP[self.gy_mask] + self.Z*self.dxV[self.gy_mask]
		self.log_dygy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_dygy_out[0,:] = self.dyP[self.gy_mask] - self.Z*self.dyV[self.gy_mask]
		self.log_dyfy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_dyfy_in[0,:] = self.dyP[self.gy_mask] + self.Z*self.dyV[self.gy_mask]
		self.log_dzgy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_dzgy_out[0,:] = self.dzP[self.gy_mask] - self.Z*self.dzV[self.gy_mask]
		self.log_dzfy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_dzfy_in[0,:] = self.dzP[self.gy_mask] + self.Z*self.dzV[self.gy_mask]

		self.log_fz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_fz_out[0,:] = self.P[self.fz_mask] + self.Z*self.W[self.fz_mask]
		self.log_gz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_gz_in[0,:] = self.P[self.fz_mask] - self.Z*self.W[self.fz_mask]
		self.log_dxfz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_dxfz_out[0,:] = self.dxP[self.fz_mask] + self.Z*self.dxW[self.fz_mask]
		self.log_dxgz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_dxgz_in[0,:] = self.dxP[self.fz_mask] - self.Z*self.dxW[self.fz_mask]
		self.log_dyfz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_dyfz_out[0,:] = self.dyP[self.fz_mask] + self.Z*self.dyW[self.fz_mask]
		self.log_dygz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_dygz_in[0,:] = self.dyP[self.fz_mask] - self.Z*self.dyW[self.fz_mask]
		self.log_dzfz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_dzfz_out[0,:] = self.dzP[self.fz_mask] + self.Z*self.dzW[self.fz_mask]
		self.log_dzgz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_dzgz_in[0,:] = self.dzP[self.fz_mask] - self.Z*self.dzW[self.fz_mask]

		self.log_gz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_gz_out[0,:] = self.P[self.gz_mask] - self.Z*self.W[self.gz_mask]
		self.log_fz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_fz_in[0,:] = self.P[self.gz_mask] + self.Z*self.W[self.gz_mask]
		self.log_dxgz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_dxgz_out[0,:] = self.dxP[self.gz_mask] - self.Z*self.dxW[self.gz_mask]
		self.log_dxfz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_dxfz_in[0,:] = self.dxP[self.gz_mask] + self.Z*self.dxW[self.gz_mask]
		self.log_dygz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_dygz_out[0,:] = self.dyP[self.gz_mask] - self.Z*self.dyW[self.gz_mask]
		self.log_dyfz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_dyfz_in[0,:] = self.dyP[self.gz_mask] + self.Z*self.dyW[self.gz_mask]
		self.log_dzgz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_dzgz_out[0,:] = self.dzP[self.gz_mask] - self.Z*self.dzW[self.gz_mask]
		self.log_dzfz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_dzfz_in[0,:] = self.dzP[self.gz_mask] + self.Z*self.dzW[self.gz_mask]


		self.static = 0

	def boundaryconditionx(self, fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx):
		self.log_gx_in = cp.roll(self.log_gx_in, 1, 0); self.log_gx_in[0,:] = gx[self.fx_mask]
		self.log_dxgx_in = cp.roll(self.log_dxgx_in, 1, 0); self.log_dxgx_in[0,:] = dxgx[self.fx_mask]
		self.log_dygx_in = cp.roll(self.log_dygx_in, 1, 0); self.log_dygx_in[0,:] = dygx[self.fx_mask]
		self.log_dzgx_in = cp.roll(self.log_dzgx_in, 1, 0); self.log_dzgx_in[0,:] = dzgx[self.fx_mask]

		fx[self.fx_mask] = -cp.sum(self.log_fx_out*self.Bs_fx, axis = 0) + cp.sum(self.log_gx_in*self.As_fx, axis = 0)
		dxfx[self.fx_mask] = -cp.sum(self.log_dxfx_out*self.Bs_fx, axis = 0) - cp.sum(self.log_dxgx_in*self.As_fx, axis = 0)
		dyfx[self.fx_mask] = -cp.sum(self.log_dyfx_out*self.Bs_fx, axis = 0) + cp.sum(self.log_dygx_in*self.As_fx, axis = 0)
		dzfx[self.fx_mask] = -cp.sum(self.log_dzfx_out*self.Bs_fx, axis = 0) + cp.sum(self.log_dzgx_in*self.As_fx, axis = 0)

		self.log_fx_out = cp.roll(self.log_fx_out, 1, 0); self.log_fx_out[0,:] = fx[self.fx_mask]
		self.log_dxfx_out = cp.roll(self.log_dxfx_out, 1, 0); self.log_dxfx_out[0,:] = dxfx[self.fx_mask]
		self.log_dyfx_out = cp.roll(self.log_dyfx_out, 1, 0); self.log_dyfx_out[0,:] = dyfx[self.fx_mask]
		self.log_dzfx_out = cp.roll(self.log_dzfx_out, 1, 0); self.log_dzfx_out[0,:] = dzfx[self.fx_mask]


		self.log_fx_in = cp.roll(self.log_fx_in, 1, 0); self.log_fx_in[0,:] = fx[self.gx_mask]
		self.log_dxfx_in = cp.roll(self.log_dxfx_in, 1, 0); self.log_dxfx_in[0,:] = dxfx[self.gx_mask]
		self.log_dyfx_in = cp.roll(self.log_dyfx_in, 1, 0); self.log_dyfx_in[0,:] = dyfx[self.gx_mask]
		self.log_dzfx_in = cp.roll(self.log_dzfx_in, 1, 0); self.log_dzfx_in[0,:] = dzfx[self.gx_mask]

		gx[self.gx_mask] = -cp.sum(self.log_gx_out*self.Bs_gx, axis = 0) + cp.sum(self.log_fx_in*self.As_gx, axis = 0)
		dxgx[self.gx_mask] = -cp.sum(self.log_dxgx_out*self.Bs_gx, axis = 0) - cp.sum(self.log_dxfx_in*self.As_gx, axis = 0)
		dygx[self.gx_mask] = -cp.sum(self.log_dygx_out*self.Bs_gx, axis = 0) + cp.sum(self.log_dyfx_in*self.As_gx, axis = 0)
		dzgx[self.gx_mask] = -cp.sum(self.log_dzgx_out*self.Bs_gx, axis = 0) + cp.sum(self.log_dzfx_in*self.As_gx, axis = 0)

		self.log_gx_out = cp.roll(self.log_gx_out, 1, 0); self.log_gx_out[0,:] = gx[self.gx_mask]
		self.log_dxgx_out = cp.roll(self.log_dxgx_out, 1, 0); self.log_dxgx_out[0,:] = dxgx[self.gx_mask]
		self.log_dygx_out = cp.roll(self.log_dygx_out, 1, 0); self.log_dygx_out[0,:] = dygx[self.gx_mask]
		self.log_dzgx_out = cp.roll(self.log_dzgx_out, 1, 0); self.log_dzgx_out[0,:] = dzgx[self.gx_mask]


		return fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx

	def boundaryconditiony(self, fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy):
		self.log_gy_in = cp.roll(self.log_gy_in, 1, 0); self.log_gy_in[0,:] = gy[self.fy_mask]
		self.log_dygy_in = cp.roll(self.log_dygy_in, 1, 0); self.log_dygy_in[0,:] = dygy[self.fy_mask]
		self.log_dxgy_in = cp.roll(self.log_dxgy_in, 1, 0); self.log_dxgy_in[0,:] = dxgy[self.fy_mask]
		self.log_dzgy_in = cp.roll(self.log_dzgy_in, 1, 0); self.log_dzgy_in[0,:] = dzgy[self.fy_mask]

		fy[self.fy_mask] = -cp.sum(self.log_fy_out*self.Bs_fy, axis = 0) + cp.sum(self.log_gy_in*self.As_fy, axis = 0)
		dyfy[self.fy_mask] = -cp.sum(self.log_dyfy_out*self.Bs_fy, axis = 0) - cp.sum(self.log_dygy_in*self.As_fy, axis = 0)
		dxfy[self.fy_mask] = -cp.sum(self.log_dxfy_out*self.Bs_fy, axis = 0) + cp.sum(self.log_dxgy_in*self.As_fy, axis = 0)
		dzfy[self.fy_mask] = -cp.sum(self.log_dzfy_out*self.Bs_fy, axis = 0) + cp.sum(self.log_dzgy_in*self.As_fy, axis = 0)

		self.log_fy_out = cp.roll(self.log_fy_out, 1, 0); self.log_fy_out[0,:] = fy[self.fy_mask]
		self.log_dyfy_out = cp.roll(self.log_dyfy_out, 1, 0); self.log_dyfy_out[0,:] = dyfy[self.fy_mask]
		self.log_dxfy_out = cp.roll(self.log_dxfy_out, 1, 0); self.log_dxfy_out[0,:] = dxfy[self.fy_mask]
		self.log_dzfy_out = cp.roll(self.log_dzfy_out, 1, 0); self.log_dzfy_out[0,:] = dzfy[self.fy_mask]

		self.log_fy_in = cp.roll(self.log_fy_in, 1, 0); self.log_fy_in[0,:] = fy[self.gy_mask]
		self.log_dyfy_in = cp.roll(self.log_dyfy_in, 1, 0); self.log_dyfy_in[0,:] = dyfy[self.gy_mask]
		self.log_dxfy_in = cp.roll(self.log_dxfy_in, 1, 0); self.log_dxfy_in[0,:] = dxfy[self.gy_mask]
		self.log_dzfy_in = cp.roll(self.log_dzfy_in, 1, 0); self.log_dzfy_in[0,:] = dzfy[self.gy_mask]

		gy[self.gy_mask] = -cp.sum(self.log_gy_out*self.Bs_gy, axis = 0) + cp.sum(self.log_fy_in*self.As_gy, axis = 0)
		dygy[self.gy_mask] = -cp.sum(self.log_dygy_out*self.Bs_gy, axis = 0) - cp.sum(self.log_dyfy_in*self.As_gy, axis = 0)
		dxgy[self.gy_mask] = -cp.sum(self.log_dxgy_out*self.Bs_gy, axis = 0) + cp.sum(self.log_dxfy_in*self.As_gy, axis = 0)
		dzgy[self.gy_mask] = -cp.sum(self.log_dzgy_out*self.Bs_gy, axis = 0) + cp.sum(self.log_dzfy_in*self.As_gy, axis = 0)

		self.log_gy_out = cp.roll(self.log_gy_out, 1, 0); self.log_gy_out[0,:] = gy[self.gy_mask]
		self.log_dygy_out = cp.roll(self.log_dygy_out, 1, 0); self.log_dygy_out[0,:] = dygy[self.gy_mask]
		self.log_dxgy_out = cp.roll(self.log_dxgy_out, 1, 0); self.log_dxgy_out[0,:] = dxgy[self.gy_mask]
		self.log_dzgy_out = cp.roll(self.log_dzgy_out, 1, 0); self.log_dzgy_out[0,:] = dzgy[self.gy_mask]

		return fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy

	def boundaryconditionz(self, fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz):
		self.log_gz_in = cp.roll(self.log_gz_in, 1, 0); self.log_gz_in[0,:] = gz[self.fz_mask]
		self.log_dygz_in = cp.roll(self.log_dygz_in, 1, 0); self.log_dygz_in[0,:] = dygz[self.fz_mask]
		self.log_dxgz_in = cp.roll(self.log_dxgz_in, 1, 0); self.log_dxgz_in[0,:] = dxgz[self.fz_mask]
		self.log_dzgz_in = cp.roll(self.log_dzgz_in, 1, 0); self.log_dzgz_in[0,:] = dzgz[self.fz_mask]

		fz[self.fz_mask] = -cp.sum(self.log_fz_out*self.Bs_fz, axis = 0) + cp.sum(self.log_gz_in*self.As_fz, axis = 0)
		dzfz[self.fz_mask] = -cp.sum(self.log_dzfz_out*self.Bs_fz, axis = 0) - cp.sum(self.log_dzgz_in*self.As_fz, axis = 0)
		dxfz[self.fz_mask] = -cp.sum(self.log_dxfz_out*self.Bs_fz, axis = 0) + cp.sum(self.log_dxgz_in*self.As_fz, axis = 0)
		dyfz[self.fz_mask] = -cp.sum(self.log_dyfz_out*self.Bs_fz, axis = 0) + cp.sum(self.log_dygz_in*self.As_fz, axis = 0)

		self.log_fz_out = cp.roll(self.log_fz_out, 1, 0); self.log_fz_out[0,:] = fz[self.fz_mask]
		self.log_dyfz_out = cp.roll(self.log_dyfz_out, 1, 0); self.log_dyfz_out[0,:] = dyfz[self.fz_mask]
		self.log_dxfz_out = cp.roll(self.log_dxfz_out, 1, 0); self.log_dxfz_out[0,:] = dxfz[self.fz_mask]
		self.log_dzfz_out = cp.roll(self.log_dzfz_out, 1, 0); self.log_dzfz_out[0,:] = dzfz[self.fz_mask]

		self.log_fz_in = cp.roll(self.log_fz_in, 1, 0); self.log_fz_in[0,:] = fz[self.gz_mask]
		self.log_dyfz_in = cp.roll(self.log_dyfz_in, 1, 0); self.log_dyfz_in[0,:] = dyfz[self.gz_mask]
		self.log_dxfz_in = cp.roll(self.log_dxfz_in, 1, 0); self.log_dxfz_in[0,:] = dxfz[self.gz_mask]
		self.log_dzfz_in = cp.roll(self.log_dzfz_in, 1, 0); self.log_dzfz_in[0,:] = dzfz[self.gz_mask]

		gz[self.gz_mask] = -cp.sum(self.log_gz_out*self.Bs_gz, axis = 0) + cp.sum(self.log_fz_in*self.As_gz, axis = 0)
		dzgz[self.gz_mask] = -cp.sum(self.log_dzgz_out*self.Bs_gz, axis = 0) - cp.sum(self.log_dzfz_in*self.As_gz, axis = 0)
		dxgz[self.gz_mask] = -cp.sum(self.log_dxgz_out*self.Bs_gz, axis = 0) + cp.sum(self.log_dxfz_in*self.As_gz, axis = 0)
		dygz[self.gz_mask] = -cp.sum(self.log_dygz_out*self.Bs_gz, axis = 0) + cp.sum(self.log_dyfz_in*self.As_gz, axis = 0)

		self.log_gz_out = cp.roll(self.log_gz_out, 1, 0); self.log_gz_out[0,:] = gz[self.gz_mask]
		self.log_dygz_out = cp.roll(self.log_dygz_out, 1, 0); self.log_dygz_out[0,:] = dygz[self.gz_mask]
		self.log_dxgz_out = cp.roll(self.log_dxgz_out, 1, 0); self.log_dxgz_out[0,:] = dxgz[self.gz_mask]
		self.log_dzgz_out = cp.roll(self.log_dzgz_out, 1, 0); self.log_dzgz_out[0,:] = dzgz[self.gz_mask]

		return fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz


"""
/******************************/
M型RCIP法による3次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class MRCIP_IIR(MCIP):
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5, alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dzp_init, dx, dt, voxel_label, As, Bs, rho, k)
		self.alpha = alpha

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (cp.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new

"""
/******************************/
C型CIP法による3次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class CCIP_IIR(MCIP_IIR):
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dxyp_init, dyzp_init, dxzp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5):
		super().__init__(p_init, dxp_init, dyp_init, dzp_init, dx, dt, voxel_label, As, Bs, rho, k)
		order = As.shape[1]
		shape = (p_init.shape[0]+2, p_init.shape[1]+2, p_init.shape[2]+2)
		self.dxyP = cp.zeros(shape); self.dxyP[1:-1,1:-1,1:-1] = dxyp_init; self.dxyP[self.wall_mask] = cp.nan
		self.dxyU = cp.zeros(shape); self.dxyU[self.wall_mask] = cp.nan
		self.dxyV = cp.zeros(shape); self.dxyV[self.wall_mask] = cp.nan
		self.dxyW = cp.zeros(shape); self.dxyW[self.wall_mask] = cp.nan

		self.dyzP = cp.zeros(shape); self.dyzP[1:-1,1:-1,1:-1] = dyzp_init; self.dyzP[self.wall_mask] = cp.nan
		self.dyzU = cp.zeros(shape); self.dyzU[self.wall_mask] = cp.nan
		self.dyzV = cp.zeros(shape); self.dyzV[self.wall_mask] = cp.nan
		self.dyzW = cp.zeros(shape); self.dyzW[self.wall_mask] = cp.nan

		self.dxzP = cp.zeros(shape); self.dxzP[1:-1,1:-1,1:-1] = dxzp_init; self.dxzP[self.wall_mask] = cp.nan
		self.dxzU = cp.zeros(shape); self.dxzU[self.wall_mask] = cp.nan
		self.dxzV = cp.zeros(shape); self.dxzV[self.wall_mask] = cp.nan
		self.dxzW = cp.zeros(shape); self.dxzW[self.wall_mask] = cp.nan

		self.log_dxyfx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_dxyfx_out[0,:] = self.dxyP[self.fx_mask] + self.Z*self.dxyU[self.fx_mask]
		self.log_dxygx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_dxygx_in[0,:] = self.dxyP[self.fx_mask] - self.Z*self.dxyU[self.fx_mask]
		self.log_dxygx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_dxygx_out[0,:] = self.dxyP[self.gx_mask] - self.Z*self.dxyU[self.gx_mask]
		self.log_dxyfx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_dxyfx_in[0,:] = self.dxyP[self.gx_mask] + self.Z*self.dxyU[self.gx_mask]
		self.log_dxyfy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_dxyfy_out[0,:] = self.dxyP[self.fy_mask] + self.Z*self.dxyV[self.fy_mask]
		self.log_dxygy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_dxygy_in[0,:] = self.dxyP[self.fy_mask] - self.Z*self.dxyV[self.fy_mask]
		self.log_dxygy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_dxygy_out[0,:] = self.dxyP[self.gy_mask] - self.Z*self.dxyV[self.gy_mask]
		self.log_dxyfy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_dxyfy_in[0,:] = self.dxyP[self.gy_mask] + self.Z*self.dxyV[self.gy_mask]
		self.log_dxyfz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_dxyfz_out[0,:] = self.dxyP[self.fz_mask] + self.Z*self.dxyW[self.fz_mask]
		self.log_dxygz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_dxygz_in[0,:] = self.dxyP[self.fz_mask] - self.Z*self.dxyW[self.fz_mask]
		self.log_dxygz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_dxygz_out[0,:] = self.dxyP[self.gz_mask] - self.Z*self.dxyW[self.gz_mask]
		self.log_dxyfz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_dxyfz_in[0,:] = self.dxyP[self.gz_mask] + self.Z*self.dxyW[self.gz_mask]

		self.log_dyzfx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_dyzfx_out[0,:] = self.dyzP[self.fx_mask] + self.Z*self.dyzU[self.fx_mask]
		self.log_dyzgx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_dyzgx_in[0,:] = self.dyzP[self.fx_mask] - self.Z*self.dyzU[self.fx_mask]
		self.log_dyzgx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_dyzgx_out[0,:] = self.dyzP[self.gx_mask] - self.Z*self.dyzU[self.gx_mask]
		self.log_dyzfx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_dyzfx_in[0,:] = self.dyzP[self.gx_mask] + self.Z*self.dyzU[self.gx_mask]
		self.log_dyzfy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_dyzfy_out[0,:] = self.dyzP[self.fy_mask] + self.Z*self.dyzV[self.fy_mask]
		self.log_dyzgy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_dyzgy_in[0,:] = self.dyzP[self.fy_mask] - self.Z*self.dyzV[self.fy_mask]
		self.log_dyzgy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_dyzgy_out[0,:] = self.dyzP[self.gy_mask] - self.Z*self.dyzV[self.gy_mask]
		self.log_dyzfy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_dyzfy_in[0,:] = self.dyzP[self.gy_mask] + self.Z*self.dyzV[self.gy_mask]
		self.log_dyzfz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_dyzfz_out[0,:] = self.dyzP[self.fz_mask] + self.Z*self.dyzW[self.fz_mask]
		self.log_dyzgz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_dyzgz_in[0,:] = self.dyzP[self.fz_mask] - self.Z*self.dyzW[self.fz_mask]
		self.log_dyzgz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_dyzgz_out[0,:] = self.dyzP[self.gz_mask] - self.Z*self.dyzW[self.gz_mask]
		self.log_dyzfz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_dyzfz_in[0,:] = self.dyzP[self.gz_mask] + self.Z*self.dyzW[self.gz_mask]

		self.log_dxzfx_out = cp.zeros((order-1, len(self.fx_mask[0])))
		self.log_dxzfx_out[0,:] = self.dxzP[self.fx_mask] + self.Z*self.dxzU[self.fx_mask]
		self.log_dxzgx_in = cp.zeros((order, len(self.fx_mask[0])))
		self.log_dxzgx_in[0,:] = self.dxzP[self.fx_mask] - self.Z*self.dxzU[self.fx_mask]
		self.log_dxzgx_out = cp.zeros((order-1, len(self.gx_mask[0])))
		self.log_dxzgx_out[0,:] = self.dxzP[self.gx_mask] - self.Z*self.dxzU[self.gx_mask]
		self.log_dxzfx_in = cp.zeros((order, len(self.gx_mask[0])))
		self.log_dxzfx_in[0,:] = self.dxzP[self.gx_mask] + self.Z*self.dxzU[self.gx_mask]
		self.log_dxzfy_out = cp.zeros((order-1, len(self.fy_mask[0])))
		self.log_dxzfy_out[0,:] = self.dxzP[self.fy_mask] + self.Z*self.dxzV[self.fy_mask]
		self.log_dxzgy_in = cp.zeros((order, len(self.fy_mask[0])))
		self.log_dxzgy_in[0,:] = self.dxzP[self.fy_mask] - self.Z*self.dxzV[self.fy_mask]
		self.log_dxzgy_out = cp.zeros((order-1, len(self.gy_mask[0])))
		self.log_dxzgy_out[0,:] = self.dxzP[self.gy_mask] - self.Z*self.dxzV[self.gy_mask]
		self.log_dxzfy_in = cp.zeros((order, len(self.gy_mask[0])))
		self.log_dxzfy_in[0,:] = self.dxzP[self.gy_mask] + self.Z*self.dxzV[self.gy_mask]
		self.log_dxzfz_out = cp.zeros((order-1, len(self.fz_mask[0])))
		self.log_dxzfz_out[0,:] = self.dxzP[self.fz_mask] + self.Z*self.dxzW[self.fz_mask]
		self.log_dxzgz_in = cp.zeros((order, len(self.fz_mask[0])))
		self.log_dxzgz_in[0,:] = self.dxzP[self.fz_mask] - self.Z*self.dxzW[self.fz_mask]
		self.log_dxzgz_out = cp.zeros((order-1, len(self.gz_mask[0])))
		self.log_dxzgz_out[0,:] = self.dxzP[self.gz_mask] - self.Z*self.dxzW[self.gz_mask]
		self.log_dxzfz_in = cp.zeros((order, len(self.gz_mask[0])))
		self.log_dxzfz_in[0,:] = self.dxzP[self.gz_mask] + self.Z*self.dxzW[self.gz_mask]

	def boundaryconditionx(self, fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx):
		self.log_gx_in = cp.roll(self.log_gx_in, 1, 0); self.log_gx_in[0,:] = gx[self.fx_mask]
		self.log_dxgx_in = cp.roll(self.log_dxgx_in, 1, 0); self.log_dxgx_in[0,:] = dxgx[self.fx_mask]
		self.log_dygx_in = cp.roll(self.log_dygx_in, 1, 0); self.log_dygx_in[0,:] = dygx[self.fx_mask]
		self.log_dzgx_in = cp.roll(self.log_dzgx_in, 1, 0); self.log_dzgx_in[0,:] = dzgx[self.fx_mask]
		self.log_dxygx_in = cp.roll(self.log_dxygx_in, 1, 0); self.log_dxygx_in[0,:] = dxygx[self.fx_mask]
		self.log_dxzgx_in = cp.roll(self.log_dxzgx_in, 1, 0); self.log_dxzgx_in[0,:] = dxzgx[self.fx_mask]

		fx[self.fx_mask] = -cp.sum(self.log_fx_out*self.Bs_fx, axis = 0) + cp.sum(self.log_gx_in*self.As_fx, axis = 0)
		dxfx[self.fx_mask] = -cp.sum(self.log_dxfx_out*self.Bs_fx, axis = 0) - cp.sum(self.log_dxgx_in*self.As_fx, axis = 0)
		dyfx[self.fx_mask] = -cp.sum(self.log_dyfx_out*self.Bs_fx, axis = 0) + cp.sum(self.log_dygx_in*self.As_fx, axis = 0)
		dzfx[self.fx_mask] = -cp.sum(self.log_dzfx_out*self.Bs_fx, axis = 0) + cp.sum(self.log_dzgx_in*self.As_fx, axis = 0)
		dxyfx[self.fx_mask] = -cp.sum(self.log_dxyfx_out*self.Bs_fx, axis = 0) - cp.sum(self.log_dxygx_in*self.As_fx, axis = 0)
		dxzfx[self.fx_mask] = -cp.sum(self.log_dxzfx_out*self.Bs_fx, axis = 0) - cp.sum(self.log_dxzgx_in*self.As_fx, axis = 0)

		self.log_fx_out = cp.roll(self.log_fx_out, 1, 0); self.log_fx_out[0,:] = fx[self.fx_mask]
		self.log_dxfx_out = cp.roll(self.log_dxfx_out, 1, 0); self.log_dxfx_out[0,:] = dxfx[self.fx_mask]
		self.log_dyfx_out = cp.roll(self.log_dyfx_out, 1, 0); self.log_dyfx_out[0,:] = dyfx[self.fx_mask]
		self.log_dzfx_out = cp.roll(self.log_dzfx_out, 1, 0); self.log_dzfx_out[0,:] = dzfx[self.fx_mask]
		self.log_dxyfx_out = cp.roll(self.log_dxyfx_out, 1, 0); self.log_dxyfx_out[0,:] = dxyfx[self.fx_mask]
		self.log_dxzfx_out = cp.roll(self.log_dxzfx_out, 1, 0); self.log_dxzfx_out[0,:] = dxzfx[self.fx_mask]

		self.log_fx_in = cp.roll(self.log_fx_in, 1, 0); self.log_fx_in[0,:] = fx[self.gx_mask]
		self.log_dxfx_in = cp.roll(self.log_dxfx_in, 1, 0); self.log_dxfx_in[0,:] = dxfx[self.gx_mask]
		self.log_dyfx_in = cp.roll(self.log_dyfx_in, 1, 0); self.log_dyfx_in[0,:] = dyfx[self.gx_mask]
		self.log_dzfx_in = cp.roll(self.log_dzfx_in, 1, 0); self.log_dzfx_in[0,:] = dzfx[self.gx_mask]
		self.log_dxyfx_in = cp.roll(self.log_dxyfx_in, 1, 0); self.log_dxyfx_in[0,:] = dxyfx[self.gx_mask]
		self.log_dxzfx_in = cp.roll(self.log_dxzfx_in, 1, 0); self.log_dxzfx_in[0,:] = dxzfx[self.gx_mask]

		gx[self.gx_mask] = -cp.sum(self.log_gx_out*self.Bs_gx, axis = 0) + cp.sum(self.log_fx_in*self.As_gx, axis = 0)
		dxgx[self.gx_mask] = -cp.sum(self.log_dxgx_out*self.Bs_gx, axis = 0) - cp.sum(self.log_dxfx_in*self.As_gx, axis = 0)
		dygx[self.gx_mask] = -cp.sum(self.log_dygx_out*self.Bs_gx, axis = 0) + cp.sum(self.log_dyfx_in*self.As_gx, axis = 0)
		dzgx[self.gx_mask] = -cp.sum(self.log_dzgx_out*self.Bs_gx, axis = 0) + cp.sum(self.log_dzfx_in*self.As_gx, axis = 0)
		dxygx[self.gx_mask] = -cp.sum(self.log_dxygx_out*self.Bs_gx, axis = 0) - cp.sum(self.log_dxyfx_in*self.As_gx, axis = 0)
		dxzgx[self.gx_mask] = -cp.sum(self.log_dxzgx_out*self.Bs_gx, axis = 0) - cp.sum(self.log_dxzfx_in*self.As_gx, axis = 0)

		self.log_gx_out = cp.roll(self.log_gx_out, 1, 0); self.log_gx_out[0,:] = gx[self.gx_mask]
		self.log_dxgx_out = cp.roll(self.log_dxgx_out, 1, 0); self.log_dxgx_out[0,:] = dxgx[self.gx_mask]
		self.log_dygx_out = cp.roll(self.log_dygx_out, 1, 0); self.log_dygx_out[0,:] = dygx[self.gx_mask]
		self.log_dzgx_out = cp.roll(self.log_dzgx_out, 1, 0); self.log_dzgx_out[0,:] = dzgx[self.gx_mask]
		self.log_dxygx_out = cp.roll(self.log_dxygx_out, 1, 0); self.log_dxygx_out[0,:] = dxygx[self.gx_mask]
		self.log_dxzgx_out = cp.roll(self.log_dxzgx_out, 1, 0); self.log_dxzgx_out[0,:] = dxzgx[self.gx_mask]

		return fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx

	def boundaryconditiony(self, fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy):
		self.log_gy_in = cp.roll(self.log_gy_in, 1, 0); self.log_gy_in[0,:] = gy[self.fy_mask]
		self.log_dygy_in = cp.roll(self.log_dygy_in, 1, 0); self.log_dygy_in[0,:] = dygy[self.fy_mask]
		self.log_dxgy_in = cp.roll(self.log_dxgy_in, 1, 0); self.log_dxgy_in[0,:] = dxgy[self.fy_mask]
		self.log_dzgy_in = cp.roll(self.log_dzgy_in, 1, 0); self.log_dzgy_in[0,:] = dzgy[self.fy_mask]
		self.log_dyzgy_in = cp.roll(self.log_dyzgy_in, 1, 0); self.log_dyzgy_in[0,:] = dyzgy[self.fy_mask]
		self.log_dxygy_in = cp.roll(self.log_dxygy_in, 1, 0); self.log_dxygy_in[0,:] = dxygy[self.fy_mask]

		fy[self.fy_mask] = -cp.sum(self.log_fy_out*self.Bs_fy, axis = 0) + cp.sum(self.log_gy_in*self.As_fy, axis = 0)
		dyfy[self.fy_mask] = -cp.sum(self.log_dyfy_out*self.Bs_fy, axis = 0) - cp.sum(self.log_dygy_in*self.As_fy, axis = 0)
		dxfy[self.fy_mask] = -cp.sum(self.log_dxfy_out*self.Bs_fy, axis = 0) + cp.sum(self.log_dxgy_in*self.As_fy, axis = 0)
		dzfy[self.fy_mask] = -cp.sum(self.log_dzfy_out*self.Bs_fy, axis = 0) + cp.sum(self.log_dzgy_in*self.As_fy, axis = 0)
		dxyfy[self.fy_mask] = -cp.sum(self.log_dxyfy_out*self.Bs_fy, axis = 0) - cp.sum(self.log_dxygy_in*self.As_fy, axis = 0)
		dyzfy[self.fy_mask] = -cp.sum(self.log_dyzfy_out*self.Bs_fy, axis = 0) - cp.sum(self.log_dyzgy_in*self.As_fy, axis = 0)

		self.log_fy_out = cp.roll(self.log_fy_out, 1, 0); self.log_fy_out[0,:] = fy[self.fy_mask]
		self.log_dyfy_out = cp.roll(self.log_dyfy_out, 1, 0); self.log_dyfy_out[0,:] = dyfy[self.fy_mask]
		self.log_dxfy_out = cp.roll(self.log_dxfy_out, 1, 0); self.log_dxfy_out[0,:] = dxfy[self.fy_mask]
		self.log_dzfy_out = cp.roll(self.log_dzfy_out, 1, 0); self.log_dzfy_out[0,:] = dzfy[self.fy_mask]
		self.log_dyzfy_out = cp.roll(self.log_dyzfy_out, 1, 0); self.log_dyzfy_out[0,:] = dyzfy[self.fy_mask]
		self.log_dxyfy_out = cp.roll(self.log_dxyfy_out, 1, 0); self.log_dxyfy_out[0,:] = dxyfy[self.fy_mask]

		self.log_fy_in = cp.roll(self.log_fy_in, 1, 0); self.log_fy_in[0,:] = fy[self.gy_mask]
		self.log_dyfy_in = cp.roll(self.log_dyfy_in, 1, 0); self.log_dyfy_in[0,:] = dyfy[self.gy_mask]
		self.log_dxfy_in = cp.roll(self.log_dxfy_in, 1, 0); self.log_dxfy_in[0,:] = dxfy[self.gy_mask]
		self.log_dzfy_in = cp.roll(self.log_dzfy_in, 1, 0); self.log_dzfy_in[0,:] = dzfy[self.gy_mask]
		self.log_dxyfy_in = cp.roll(self.log_dxyfy_in, 1, 0); self.log_dxyfy_in[0,:] = dxyfy[self.gy_mask]
		self.log_dyzfy_in = cp.roll(self.log_dyzfy_in, 1, 0); self.log_dyzfy_in[0,:] = dyzfy[self.gy_mask]

		gy[self.gy_mask] = -cp.sum(self.log_gy_out*self.Bs_gy, axis = 0) + cp.sum(self.log_fy_in*self.As_gy, axis = 0)
		dygy[self.gy_mask] = -cp.sum(self.log_dygy_out*self.Bs_gy, axis = 0) - cp.sum(self.log_dyfy_in*self.As_gy, axis = 0)
		dxgy[self.gy_mask] = -cp.sum(self.log_dxgy_out*self.Bs_gy, axis = 0) + cp.sum(self.log_dxfy_in*self.As_gy, axis = 0)
		dzgy[self.gy_mask] = -cp.sum(self.log_dzgy_out*self.Bs_gy, axis = 0) + cp.sum(self.log_dzfy_in*self.As_gy, axis = 0)
		dxygy[self.gy_mask] = -cp.sum(self.log_dxygy_out*self.Bs_gy, axis = 0) - cp.sum(self.log_dxyfy_in*self.As_gy, axis = 0)
		dyzgy[self.gy_mask] = -cp.sum(self.log_dyzgy_out*self.Bs_gy, axis = 0) - cp.sum(self.log_dyzfy_in*self.As_gy, axis = 0)

		self.log_gy_out = cp.roll(self.log_gy_out, 1, 0); self.log_gy_out[0,:] = gy[self.gy_mask]
		self.log_dygy_out = cp.roll(self.log_dygy_out, 1, 0); self.log_dygy_out[0,:] = dygy[self.gy_mask]
		self.log_dxgy_out = cp.roll(self.log_dxgy_out, 1, 0); self.log_dxgy_out[0,:] = dxgy[self.gy_mask]
		self.log_dzgy_out = cp.roll(self.log_dzgy_out, 1, 0); self.log_dzgy_out[0,:] = dzgy[self.gy_mask]
		self.log_dxygy_out = cp.roll(self.log_dxygy_out, 1, 0); self.log_dxygy_out[0,:] = dxygy[self.gy_mask]
		self.log_dyzgy_out = cp.roll(self.log_dyzgy_out, 1, 0); self.log_dyzgy_out[0,:] = dyzgy[self.gy_mask]

		return fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy

	def boundaryconditionz(self, fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz):
		self.log_gz_in = cp.roll(self.log_gz_in, 1, 0); self.log_gz_in[0,:] = gz[self.fz_mask]
		self.log_dygz_in = cp.roll(self.log_dygz_in, 1, 0); self.log_dygz_in[0,:] = dygz[self.fz_mask]
		self.log_dxgz_in = cp.roll(self.log_dxgz_in, 1, 0); self.log_dxgz_in[0,:] = dxgz[self.fz_mask]
		self.log_dzgz_in = cp.roll(self.log_dzgz_in, 1, 0); self.log_dzgz_in[0,:] = dzgz[self.fz_mask]
		self.log_dxzgz_in = cp.roll(self.log_dxzgz_in, 1, 0); self.log_dxzgz_in[0,:] = dxzgz[self.fz_mask]
		self.log_dyzgz_in = cp.roll(self.log_dyzgz_in, 1, 0); self.log_dyzgz_in[0,:] = dyzgz[self.fz_mask]

		fz[self.fz_mask] = -cp.sum(self.log_fz_out*self.Bs_fz, axis = 0) + cp.sum(self.log_gz_in*self.As_fz, axis = 0)
		dzfz[self.fz_mask] = -cp.sum(self.log_dzfz_out*self.Bs_fz, axis = 0) - cp.sum(self.log_dzgz_in*self.As_fz, axis = 0)
		dxfz[self.fz_mask] = -cp.sum(self.log_dxfz_out*self.Bs_fz, axis = 0) + cp.sum(self.log_dxgz_in*self.As_fz, axis = 0)
		dyfz[self.fz_mask] = -cp.sum(self.log_dyfz_out*self.Bs_fz, axis = 0) + cp.sum(self.log_dygz_in*self.As_fz, axis = 0)
		dxzfz[self.fz_mask] = -cp.sum(self.log_dxzfz_out*self.Bs_fz, axis = 0) - cp.sum(self.log_dxzgz_in*self.As_fz, axis = 0)
		dyzfz[self.fz_mask] = -cp.sum(self.log_dyzfz_out*self.Bs_fz, axis = 0) - cp.sum(self.log_dyzgz_in*self.As_fz, axis = 0)

		self.log_fz_out = cp.roll(self.log_fz_out, 1, 0); self.log_fz_out[0,:] = fz[self.fz_mask]
		self.log_dyfz_out = cp.roll(self.log_dyfz_out, 1, 0); self.log_dyfz_out[0,:] = dyfz[self.fz_mask]
		self.log_dxfz_out = cp.roll(self.log_dxfz_out, 1, 0); self.log_dxfz_out[0,:] = dxfz[self.fz_mask]
		self.log_dzfz_out = cp.roll(self.log_dzfz_out, 1, 0); self.log_dzfz_out[0,:] = dzfz[self.fz_mask]
		self.log_dxzfz_out = cp.roll(self.log_dxzfz_out, 1, 0); self.log_dxzfz_out[0,:] = dxzfz[self.fz_mask]
		self.log_dyzfz_out = cp.roll(self.log_dyzfz_out, 1, 0); self.log_dyzfz_out[0,:] = dyzfz[self.fz_mask]

		self.log_fz_in = cp.roll(self.log_fz_in, 1, 0); self.log_fz_in[0,:] = fz[self.gz_mask]
		self.log_dyfz_in = cp.roll(self.log_dyfz_in, 1, 0); self.log_dyfz_in[0,:] = dyfz[self.gz_mask]
		self.log_dxfz_in = cp.roll(self.log_dxfz_in, 1, 0); self.log_dxfz_in[0,:] = dxfz[self.gz_mask]
		self.log_dzfz_in = cp.roll(self.log_dzfz_in, 1, 0); self.log_dzfz_in[0,:] = dzfz[self.gz_mask]
		self.log_dxzfz_in = cp.roll(self.log_dxzfz_in, 1, 0); self.log_dxzfz_in[0,:] = dxzfz[self.gz_mask]
		self.log_dyzfz_in = cp.roll(self.log_dyzfz_in, 1, 0); self.log_dyzfz_in[0,:] = dyzfz[self.gz_mask]

		gz[self.gz_mask] = -cp.sum(self.log_gz_out*self.Bs_gz, axis = 0) + cp.sum(self.log_fz_in*self.As_gz, axis = 0)
		dzgz[self.gz_mask] = -cp.sum(self.log_dzgz_out*self.Bs_gz, axis = 0) - cp.sum(self.log_dzfz_in*self.As_gz, axis = 0)
		dxgz[self.gz_mask] = -cp.sum(self.log_dxgz_out*self.Bs_gz, axis = 0) + cp.sum(self.log_dxfz_in*self.As_gz, axis = 0)
		dygz[self.gz_mask] = -cp.sum(self.log_dygz_out*self.Bs_gz, axis = 0) + cp.sum(self.log_dyfz_in*self.As_gz, axis = 0)
		dxzgz[self.gz_mask] = -cp.sum(self.log_dxzgz_out*self.Bs_gz, axis = 0) - cp.sum(self.log_dxzfz_in*self.As_gz, axis = 0)
		dyzgz[self.gz_mask] = -cp.sum(self.log_dyzgz_out*self.Bs_gz, axis = 0) - cp.sum(self.log_dyzfz_in*self.As_gz, axis = 0)

		self.log_gz_out = cp.roll(self.log_gz_out, 1, 0); self.log_gz_out[0,:] = gz[self.gz_mask]
		self.log_dygz_out = cp.roll(self.log_dygz_out, 1, 0); self.log_dygz_out[0,:] = dygz[self.gz_mask]
		self.log_dxgz_out = cp.roll(self.log_dxgz_out, 1, 0); self.log_dxgz_out[0,:] = dxgz[self.gz_mask]
		self.log_dzgz_out = cp.roll(self.log_dzgz_out, 1, 0); self.log_dzgz_out[0,:] = dzgz[self.gz_mask]
		self.log_dxzgz_out = cp.roll(self.log_dxzgz_out, 1, 0); self.log_dxzgz_out[0,:] = dxzgz[self.gz_mask]
		self.log_dyzgz_out = cp.roll(self.log_dyzgz_out, 1, 0); self.log_dyzgz_out[0,:] = dyzgz[self.gz_mask]

		return fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz

	def updatex(self):
		##########x方向の移流
		#####fx, dxfx, dyfx, dzfxの定義
		fx = self.P + self.Z*self.U
		dxfx = self.dxP + self.Z*self.dxU
		dyfx = self.dyP + self.Z*self.dyU
		dzfx = self.dzP + self.Z*self.dzU
		dxyfx = self.dxyP + self.Z*self.dxyU
		dxzfx = self.dxzP + self.Z*self.dxzU

		#####fx, dxfxの更新
		fx[1:,:,:], dxfx[1:,:,:] = self.interpolate(fx[1:,:,:], dxfx[1:,:,:], fx[:-1,:,:], dxfx[:-1,:,:], -self.dx, -self.epsilon)

		#####dyfx, dzfxの更新
		dyfx[1:,:,:], dxyfx[1:,:,:] = self.interpolate(dyfx[1:,:,:], dxyfx[1:,:,:], dyfx[:-1,:,:], dxyfx[:-1,:,:], -self.dx, -self.epsilon)
		dzfx[1:,:,:], dxzfx[1:,:,:] = self.interpolate(dzfx[1:,:,:], dxzfx[1:,:,:], dzfx[:-1,:,:], dxzfx[:-1,:,:], -self.dx, -self.epsilon)

		#####gx, dxgx, dygx, dzgxの定義
		gx = self.P - self.Z*self.U
		dxgx = self.dxP - self.Z*self.dxU
		dygx = self.dyP - self.Z*self.dyU
		dzgx = self.dzP - self.Z*self.dzU
		dxygx = self.dxyP - self.Z*self.dxyU
		dxzgx = self.dxzP - self.Z*self.dxzU

		#####gx, dxgxの更新
		gx[:-1,:,:], dxgx[:-1,:,:] = self.interpolate(gx[:-1,:,:], dxgx[:-1,:,:], gx[1:,:,:], dxgx[1:,:,:], self.dx, self.epsilon)

		#####dygxの更新
		dygx[:-1,:,:], dxygx[:-1,:,:] = self.interpolate(dygx[:-1,:,:], dxygx[:-1,:,:], dygx[1:,:,:], dxygx[1:,:,:], self.dx, self.epsilon)
		dzgx[:-1,:,:], dxzgx[:-1,:,:] = self.interpolate(dzgx[:-1,:,:], dxzgx[:-1,:,:], dzgx[1:,:,:], dxzgx[1:,:,:], self.dx, self.epsilon)

		#####境界条件
		fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx = self.boundaryconditionx(fx, gx, dxfx, dxgx, dyfx, dygx, dzfx, dzgx, dxyfx, dxygx, dxzfx, dxzgx)
		

		##########物理量の更新
		self.P = (fx + gx)/2.; self.P[self.wall_mask] = cp.nan
		self.U = (fx - gx)/(2.*self.Z); self.U[self.wall_mask] = cp.nan 
		self.dxP = (dxfx + dxgx)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxU = (dxfx - dxgx)/(2.*self.Z); self.dxU[self.wall_mask] = cp.nan
		self.dyP = (dyfx + dygx)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyU = (dyfx - dygx)/(2.*self.Z); self.dyU[self.wall_mask] = cp.nan
		self.dzP = (dzfx + dzgx)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzU = (dzfx - dzgx)/(2.*self.Z); self.dzU[self.wall_mask] = cp.nan
		self.dxyP = (dxyfx + dxygx)/2.; self.dxyP[self.wall_mask] = cp.nan
		self.dxyU = (dxyfx - dxygx)/(2.*self.Z); self.dxyU[self.wall_mask] = cp.nan
		self.dxzP = (dxzfx + dxzgx)/2.; self.dxzP[self.wall_mask] = cp.nan
		self.dxzU = (dxzfx - dxzgx)/(2.*self.Z); self.dxzU[self.wall_mask] = cp.nan



	def updatey(self):
		##########y方向の移流
		#####fy, dxfy, dyfy, dzfyの定義
		fy = self.P + self.Z*self.V
		dxfy = self.dxP + self.Z*self.dxV
		dyfy = self.dyP + self.Z*self.dyV
		dzfy = self.dzP + self.Z*self.dzV
		dxyfy = self.dxyP + self.Z*self.dxyV
		dyzfy = self.dyzP + self.Z*self.dyzV
		#####fy, dyfyの更新
		fy[:,1:,:], dyfy[:,1:,:] = self.interpolate(fy[:,1:,:], dyfy[:,1:,:], fy[:,:-1,:], dyfy[:,:-1,:], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfy[:,1:,:], dxyfy[:,1:,:] = self.interpolate(dxfy[:,1:,:], dxyfy[:,1:,:], dxfy[:,:-1,:], dxyfy[:,:-1,:], -self.dx, -self.epsilon)
		dzfy[:,1:,:], dyzfy[:,1:,:] = self.interpolate(dzfy[:,1:,:], dyzfy[:,1:,:], dzfy[:,:-1,:], dyzfy[:,:-1,:], -self.dx, -self.epsilon)

		#####gy, dxgy, dygyの定義
		gy = self.P - self.Z*self.V
		dxgy = self.dxP - self.Z*self.dxV
		dygy = self.dyP - self.Z*self.dyV
		dzgy = self.dzP - self.Z*self.dzV
		dxygy = self.dxyP - self.Z*self.dxyV
		dyzgy = self.dyzP - self.Z*self.dyzV
		#####gy, dygyの更新
		gy[:,:-1,:], dygy[:,:-1,:] = self.interpolate(gy[:,:-1,:], dygy[:,:-1,:], gy[:,1:,:], dygy[:,1:,:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgy[:,:-1,:], dxygy[:,:-1,:] = self.interpolate(dxgy[:,:-1,:], dxygy[:,:-1,:], dxgy[:,1:,:], dxygy[:,1:,:], self.dx, self.epsilon)
		dzgy[:,:-1,:], dyzgy[:,:-1,:] = self.interpolate(dzgy[:,:-1,:], dyzgy[:,:-1,:], dzgy[:,1:,:], dyzgy[:,1:,:], self.dx, self.epsilon)

		#####境界条件
		fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy = self.boundaryconditiony(fy, gy, dyfy, dygy, dxfy, dxgy, dzfy, dzgy, dxyfy, dxygy, dyzfy, dyzgy)

		##########物理量の更新
		self.P = (fy + gy)/2.; self.P[self.wall_mask] = cp.nan
		self.V = (fy - gy)/(2.*self.Z); self.V[self.wall_mask] = cp.nan 
		self.dxP = (dxfy + dxgy)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxV = (dxfy - dxgy)/(2.*self.Z); self.dxV[self.wall_mask] = cp.nan
		self.dyP = (dyfy + dygy)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyV = (dyfy - dygy)/(2.*self.Z); self.dyV[self.wall_mask] = cp.nan
		self.dzP = (dzfy + dzgy)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzV = (dzfy - dzgy)/(2.*self.Z); self.dzV[self.wall_mask] = cp.nan
		self.dxyP = (dxyfy + dxygy)/2.; self.dxyP[self.wall_mask] = cp.nan
		self.dxyV = (dxyfy - dxygy)/(2.*self.Z); self.dxyV[self.wall_mask] = cp.nan
		self.dyzP = (dyzfy + dyzgy)/2.; self.dyzP[self.wall_mask] = cp.nan
		self.dyzV = (dyzfy - dyzgy)/(2.*self.Z); self.dyzV[self.wall_mask] = cp.nan

	def updatez(self):
		##########z方向の移流
		#####fz, dxfz, dyfz, dzfzの定義
		fz = self.P + self.Z*self.W
		dxfz = self.dxP + self.Z*self.dxW
		dyfz = self.dyP + self.Z*self.dyW
		dzfz = self.dzP + self.Z*self.dzW
		dxzfz = self.dxzP + self.Z*self.dxzW
		dyzfz = self.dyzP + self.Z*self.dyzW
		#####fy, dyfyの更新
		fz[:,:,1:], dzfz[:,:,1:] = self.interpolate(fz[:,:,1:], dzfz[:,:,1:], fz[:,:,:-1], dzfz[:,:,:-1], -self.dx, -self.epsilon)

		#####dxfyの更新
		dxfz[:,:,1:], dxzfz[:,:,1:] = self.interpolate(dxfz[:,:,1:], dxzfz[:,:,1:], dxfz[:,:,:-1], dxzfz[:,:,:-1], -self.dx, -self.epsilon)
		dyfz[:,:,1:], dyzfz[:,:,1:] = self.interpolate(dyfz[:,:,1:], dyzfz[:,:,1:], dyfz[:,:,:-1], dyzfz[:,:,:-1], -self.dx, -self.epsilon)

		#####gy, dxgy, dygyの定義
		gz = self.P - self.Z*self.W
		dxgz = self.dxP - self.Z*self.dxW
		dygz = self.dyP - self.Z*self.dyW
		dzgz = self.dzP - self.Z*self.dzW
		dxzgz = self.dxzP - self.Z*self.dxzW
		dyzgz = self.dyzP - self.Z*self.dyzW
		#####gy, dygyの更新
		gz[:,:,:-1], dzgz[:,:,:-1] = self.interpolate(gz[:,:,:-1], dzgz[:,:,:-1], gz[:,:,1:], dzgz[:,:,1:], self.dx, self.epsilon)

		#####dxgyの更新
		dxgz[:,:,:-1], dxzgz[:,:,:-1] = self.interpolate(dxgz[:,:,:-1], dxzgz[:,:,:-1], dxgz[:,:,1:], dxzgz[:,:,1:], self.dx, self.epsilon)
		dygz[:,:,:-1], dyzgz[:,:,:-1] = self.interpolate(dygz[:,:,:-1], dyzgz[:,:,:-1], dygz[:,:,1:], dyzgz[:,:,1:], self.dx, self.epsilon)

		#####境界条件
		fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz = self.boundaryconditionz(fz, gz, dzfz, dzgz, dxfz, dxgz, dyfz, dygz, dxzfz, dxzgz, dyzfz, dyzgz)
		
		##########物理量の更新
		self.P = (fz + gz)/2.; self.P[self.wall_mask] = cp.nan
		self.W = (fz - gz)/(2.*self.Z); self.W[self.wall_mask] = cp.nan 
		self.dxP = (dxfz + dxgz)/2.; self.dxP[self.wall_mask] = cp.nan
		self.dxW = (dxfz - dxgz)/(2.*self.Z); self.dxW[self.wall_mask] = cp.nan
		self.dyP = (dyfz + dygz)/2.; self.dyP[self.wall_mask] = cp.nan
		self.dyW = (dyfz - dygz)/(2.*self.Z); self.dyW[self.wall_mask] = cp.nan
		self.dzP = (dzfz + dzgz)/2.; self.dzP[self.wall_mask] = cp.nan
		self.dzW = (dzfz - dzgz)/(2.*self.Z); self.dzW[self.wall_mask] = cp.nan
		self.dxzP = (dxzfz + dxzgz)/2.; self.dxzP[self.wall_mask] = cp.nan
		self.dxzW = (dxzfz - dxzgz)/(2.*self.Z); self.dxzW[self.wall_mask] = cp.nan
		self.dyzP = (dyzfz + dyzgz)/2.; self.dyzP[self.wall_mask] = cp.nan
		self.dyzW = (dyzfz - dyzgz)/(2.*self.Z); self.dyzW[self.wall_mask] = cp.nan

"""
/******************************/
C型RCIP法による3次元音伝播シミュレーション：IIRによる周波数依存B.C.
/******************************/
"""
class CRCIP_IIR(MCIP_IIR):
	def __init__(self, p_init, dxp_init, dyp_init, dzp_init, dxyp_init, dyzp_init, dxzp_init, dx, dt, voxel_label, As, Bs, rho = 1.293, k = 1.4e+5, alpha = 1.):
		super().__init__(p_init, dxp_init, dyp_init, dzp_init, dxyp_init, dyzp_init, dxzp_init, dx, dt, voxel_label, As, Bs, rho, k)
		self.alpha = 1.

	def interpolate(self, f, df, fup, dfup, D, epsilon):
		S = (fup-f)/D
		B = (cp.abs((S-df)/(dfup-S+1e-10))-1.)/D + 1e-10
		c = df + f*self.alpha*B
		a = (df-S+(dfup-S)*(1.+self.alpha*B*D))/(D**2)
		b = S*self.alpha*B + (S-df)/D - a*D

		f_new = (a*(epsilon**3)+b*(epsilon**2)+c*epsilon+f)/(1.+self.alpha*B*epsilon)
		df_new = (3.*a*(epsilon**2)+2.*b*epsilon+c-f_new*self.alpha*B)/(1.+self.alpha*B*epsilon)

		return f_new, df_new
