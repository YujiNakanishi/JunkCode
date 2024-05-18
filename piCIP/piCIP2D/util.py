import numpy as np
import math

def Label2Mask(voxel_label, Rs):
	fx_mask = np.where((np.roll(voxel_label, 1, axis = 0) > 0)*(voxel_label == 0))
	gx_mask = np.where((np.roll(voxel_label, -1, axis = 0) > 0)*(voxel_label == 0))
	fy_mask = np.where((np.roll(voxel_label, 1, axis = 1) > 0)*(voxel_label == 0))
	gy_mask = np.where((np.roll(voxel_label, -1, axis = 1) > 0)*(voxel_label == 0))
	wall_mask = np.where(voxel_label > 0)

	r_fx_label = np.roll(voxel_label, 1, axis = 0)[fx_mask]
	r_fx = np.array([Rs[rfl-1] for rfl in r_fx_label])

	r_gx_label = np.roll(voxel_label, -1, axis = 0)[gx_mask]
	r_gx = np.array([Rs[rgl-1] for rgl in r_gx_label])

	r_fy_label = np.roll(voxel_label, 1, axis = 1)[fy_mask]
	r_fy = np.array([Rs[rfl-1] for rfl in r_fy_label])

	r_gy_label = np.roll(voxel_label, -1, axis = 1)[gy_mask]
	r_gy = np.array([Rs[rgl-1] for rgl in r_gy_label])

	return fx_mask, gx_mask, fy_mask, gy_mask, wall_mask, r_fx, r_gx, r_fy, r_gy



def Label2Mask_IIR(voxel_label, As, Bs):
	fx_mask = np.where((np.roll(voxel_label, 1, axis = 0) > 0)*(voxel_label == 0))
	gx_mask = np.where((np.roll(voxel_label, -1, axis = 0) > 0)*(voxel_label == 0))
	fy_mask = np.where((np.roll(voxel_label, 1, axis = 1) > 0)*(voxel_label == 0))
	gy_mask = np.where((np.roll(voxel_label, -1, axis = 1) > 0)*(voxel_label == 0))
	wall_mask = np.where(voxel_label > 0)
	wall_mask = np.where(voxel_label > 0)

	r_fx_label = np.roll(voxel_label, 1, axis = 0)[fx_mask]
	As_fx = np.stack([As[rfl-1] for rfl in r_fx_label], axis = 1)
	Bs_fx = np.stack([Bs[rfl-1] for rfl in r_fx_label], axis = 1)

	r_gx_label = np.roll(voxel_label, -1, axis = 0)[gx_mask]
	As_gx = np.stack([As[rgl-1] for rgl in r_gx_label], axis = 1)
	Bs_gx = np.stack([Bs[rgl-1] for rgl in r_gx_label], axis = 1)

	r_fy_label = np.roll(voxel_label, 1, axis = 1)[fy_mask]
	As_fy = np.stack([As[rfl-1] for rfl in r_fy_label], axis = 1)
	Bs_fy = np.stack([Bs[rfl-1] for rfl in r_fy_label], axis = 1)

	r_gy_label = np.roll(voxel_label, -1, axis = 1)[gy_mask]
	As_gy = np.stack([As[rgl-1] for rgl in r_gy_label], axis = 1)
	Bs_gy = np.stack([Bs[rgl-1] for rgl in r_gy_label], axis = 1)

	return fx_mask, gx_mask, fy_mask, gy_mask, wall_mask, As_fx, Bs_fx, As_gx, Bs_gx, As_fy, Bs_fy, As_gy, Bs_gy