import numpy as np

def voxel2mask(voxel_label):
	wall_label = (voxel_label > 0)

	need_right_idx_x, need_right_idx_y = np.where(wall_label*(np.roll(wall_label, -1, axis = 0) == False))
	target_right_idx_x = need_right_idx_x+1

	need_left_idx_x, need_left_idx_y = np.where(wall_label*(np.roll(wall_label, 1, axis = 0) == False))
	target_left_idx_x = need_left_idx_x-1
	
	need_bottom_idx_x, need_bottom_idx_y = np.where(wall_label*(np.roll(wall_label, 1, axis = 1) == False))
	target_bottom_idx_y = need_bottom_idx_y-1
	
	need_top_idx_x, need_top_idx_y = np.where(wall_label*(np.roll(wall_label, -1, axis = 1) == False))
	target_top_idx_y = need_top_idx_y+1

	return need_left_idx_x, need_left_idx_y, target_left_idx_x, \
	need_right_idx_x, need_right_idx_y, target_right_idx_x, \
	need_bottom_idx_x, need_bottom_idx_y, target_bottom_idx_y, \
	need_top_idx_x, need_top_idx_y, target_top_idx_y