import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
import sys

class Net(nn.Module):
    def __init__(self, particle_radius = 0.025, radius_scale = 1.5):
        super().__init__()
        self.dt = 1./50.
        self.gravity = torch.tensor([[0., -9.8, 0.]]).to("cuda")
        self.extent = torch.tensor(6.*radius_scale*particle_radius).to("cuda")

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        self.cconv_wall_1 = ml3d.layers.ContinuousConv(in_channels = 3, filters = 32, kernel_size = [4, 4, 4], activation = F.relu, coordinate_mapping = "ball_to_cube_volume_preserving", normalize = False, window_function=window_poly6, radius_search_ignore_query_points=True)
        self.cconv_fluid_1 = ml3d.layers.ContinuousConv(in_channels = 3, filters = 32, kernel_size = [4, 4, 4], activation = F.relu, coordinate_mapping = "ball_to_cube_volume_preserving", normalize = False, window_function=window_poly6, radius_search_ignore_query_points=True)
        self.dense_1 = nn.Linear(in_features = 3, out_features=32)

        self.cconv_2 = ml3d.layers.ContinuousConv(in_channels = 96, filters = 64, kernel_size = [4, 4, 4], coordinate_mapping = "ball_to_cube_volume_preserving", normalize = False, window_function=window_poly6, radius_search_ignore_query_points=True)
        self.dense_2 = nn.Linear(in_features = 96, out_features = 64)

        self.cconv_3 = ml3d.layers.ContinuousConv(in_channels = 64, filters = 64, kernel_size = [4, 4, 4], coordinate_mapping = "ball_to_cube_volume_preserving", normalize = False, window_function=window_poly6, radius_search_ignore_query_points=True)
        self.dense_3 = nn.Linear(in_features = 64, out_features = 64)

        self.cconv_4 = ml3d.layers.ContinuousConv(in_channels = 64, filters = 3, kernel_size = [4, 4, 4], coordinate_mapping = "ball_to_cube_volume_preserving", normalize = False, window_function=window_poly6, radius_search_ignore_query_points=True)
        self.dense_4 = nn.Linear(in_features = 64, out_features = 3)
    
    def neighbor_num(self, pos, radius):
        num = torch.zeros(len(pos))
        for idx, p in enumerate(pos):
            distance = torch.norm(pos -p, 2, dim = 1)
            num[idx] = torch.sum(distance < radius)
        
        return num



    def forward(self, fluid_pos, wall_pos, fluid_vel, wall_normal_vec):
        wall_1 = self.cconv_wall_1(wall_normal_vec, wall_pos, fluid_pos, self.extent) #(N, F)
        fluid_1 = self.cconv_fluid_1(fluid_vel, fluid_pos, fluid_pos, self.extent) #(N, F)
        dense_1 = self.dense_1(fluid_vel)

        out_1 = torch.cat((wall_1, fluid_1, dense_1), axis = -1)

        x = self.cconv_2(out_1, fluid_pos, fluid_pos, self.extent)
        y = self.dense_2(out_1)
        out_2 = F.relu(x + y)

        x = self.cconv_3(out_2, fluid_pos, fluid_pos, self.extent)
        y = self.dense_3(out_2)
        out_3 = F.relu(out_2 + x + y)

        out_4 = self.cconv_4(out_3, fluid_pos, fluid_pos, self.extent) + self.dense_4(out_3)

        return out_4
    
    def step(self, fluid_pos, wall_pos, fluid_vel, wall_normal_vec):
        fluid_vel_internal = fluid_vel + self.gravity*self.dt
        fluid_pos_internal = fluid_pos + 0.5*(fluid_vel + fluid_vel_internal)*self.dt

        dx = self.forward(fluid_pos_internal, wall_pos, fluid_vel_internal, wall_normal_vec)

        fluid_pos_next = fluid_pos_internal + dx
        fluid_vel_next = (fluid_pos_next - fluid_pos)/self.dt

        neighbor_num = self.neighbor_num(fluid_pos_internal, 4.5*0.025*2.)

        return fluid_pos_next, fluid_vel_next, neighbor_num






# class MyParticleNetwork(torch.nn.Module):
#     def __init__(
#         self,
#         kernel_size=[4, 4, 4],
#         radius_scale=1.5,
#         coordinate_mapping='ball_to_cube_volume_preserving',
#         interpolation='linear',
#         use_window=True,
#         particle_radius=0.025,
#         timestep=1 / 50,
#         gravity=(0, -9.81, 0),
#         other_feats_channels=0,
#     ):
        
#         super().__init__()
#         self.layer_channels = [32, 64, 64, 3]
#         self.kernel_size = kernel_size
#         self.radius_scale = radius_scale
#         self.coordinate_mapping = coordinate_mapping
#         self.interpolation = interpolation
#         self.use_window = use_window
#         self.particle_radius = particle_radius
#         self.filter_extent = np.float32(self.radius_scale * 6 *self.particle_radius)
#         self.timestep = timestep
#         gravity = torch.FloatTensor(gravity)
#         self.register_buffer('gravity', gravity)

#         self._all_convs = []
        
#         def Conv(name, activation=None, **kwargs):
#             conv_fn = ml3d.layers.ContinuousConv
        
#             window_fn = None
#             if self.use_window == True:
#                 window_fn = window_poly6
            
#             conv = conv_fn(kernel_size=self.kernel_size, activation=activation, align_corners=True, interpolation=self.interpolation, coordinate_mapping=self.coordinate_mapping,
#                            normalize=False, window_function=window_fn, radius_search_ignore_query_points=True, **kwargs)
            
#             self._all_convs.append((name, conv))
#             return conv
        
#         self.conv0_fluid = Conv(name="conv0_fluid", in_channels=4 + other_feats_channels, filters=self.layer_channels[0], activation=None)
#         self.conv0_obstacle = Conv(name="conv0_obstacle", in_channels=3, filters=self.layer_channels[0], activation=None)
#         self.dense0_fluid = torch.nn.Linear(in_features=4 + other_feats_channels, out_features=self.layer_channels[0])
#         torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
#         torch.nn.init.zeros_(self.dense0_fluid.bias)

#         self.convs = []
#         self.denses = []
#         for i in range(1, len(self.layer_channels)):
#             in_ch = self.layer_channels[i - 1]
#             if i == 1:
#                 in_ch *= 3
#             out_ch = self.layer_channels[i]
#             dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
#             torch.nn.init.xavier_uniform_(dense.weight)
#             torch.nn.init.zeros_(dense.bias)

#             setattr(self, 'dense{0}'.format(i), dense)
#             conv = Conv(name='conv{0}'.format(i), in_channels=in_ch, filters=out_ch, activation=None)
    

#     def integrate_pos_vel(self, pos1, vel1):
#         """Apply gravity and integrate position and velocity"""
#         dt = self.timestep
#         vel2 = vel1 + dt * self.gravity
#         pos2 = pos1 + dt * (vel2 + vel1) / 2
#         return pos2, vel2
    
#     def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
#         """Apply the correction
#         pos1,vel1 are the positions and velocities from the previous timestep
#         pos2,vel2 are the positions after applying gravity and the integration step
#         """
#         dt = self.timestep
#         pos = pos2 + pos_correction
#         vel = (pos - pos1) / dt
#         return pos, vel
    
#     def compute_correction(self, pos, vel, other_feats, box, box_feats, fixed_radius_search_hash_table=None):
#         """Expects that the pos and vel has already been updated with gravity and velocity"""
#         # compute the extent of the filters (the diameter)
#         filter_extent = torch.tensor(self.filter_extent)
#         fluid_feats = [torch.ones_like(pos[:, 0:1]), vel]
#         if not other_feats is None:
#             fluid_feats.append(other_feats)
#         fluid_feats = torch.cat(fluid_feats, axis=-1)
        
#         self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos, filter_extent)
#         self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
#         self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos, filter_extent)

#         feats = torch.cat([self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid], axis=-1)
#         self.ans_convs = [feats]
#         for conv, dense in zip(self.convs, self.denses):
#             inp_feats = F.relu(self.ans_convs[-1])
#             ans_conv = conv(inp_feats, pos, pos, filter_extent)
#             ans_dense = dense(inp_feats)

#             if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
#                 ans = ans_conv + ans_dense + self.ans_convs[-1]
#             else:
#                 ans = ans_conv + ans_dense
#             self.ans_convs.append(ans)

#         # compute the number of fluid neighbors.
#         # this info is used in the loss function during training.
#         self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(torch.ones_like(self.conv0_fluid.nns.neighbors_index, dtype=torch.float32), self.conv0_fluid.nns.neighbors_row_splits)
#         self.last_features = self.ans_convs[-2]

#         # scale to better match the scale of the output distribution
#         self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
#         return self.pos_correction


#     def forward(self, inputs, fixed_radius_search_hash_table=None):
#         """computes 1 simulation timestep
#         inputs: list or tuple with (pos,vel,feats,box,box_feats)
#           pos and vel are the positions and velocities of the fluid particles.
#           feats is reserved for passing additional features, use None here.
#           box are the positions of the static particles and box_feats are the
#           normals of the static particles.
#         """
#         pos, vel, feats, box, box_feats = inputs

#         pos2, vel2 = self.integrate_pos_vel(pos, vel)
#         pos_correction = self.compute_correction(
#             pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
#         pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
#             pos, vel, pos2, vel2, pos_correction)

#         return pos2_corrected, vel2_corrected