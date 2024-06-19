from module import dataset, Networks
from module import priFunctool as pri
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyvista as pv
import sys
import copy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

data_list, _ = dataset.train_test_split()
dataset = dataset.RelDistance(data_list)

#####正解結果および補間結果の出力
fem_data = dataset.data_list[0]
geometry = pv.read(fem_data["file_dir"] + "/{}mm.vtk".format(fem_data["HR_scale"]))

x_ans = geometry.points
x_ans_0 = copy.deepcopy(x_ans[0])
x_ans -= x_ans_0 #x_predと同様にx_0を(0, 0, 0)に拘束
geometry.points = x_ans
geometry.save("ans.vtk") #正解形状の出力
x_interp = np.load(fem_data["file_dir"] + "/x_interp_{0}mm_{1}mm.npy".format(fem_data["HR_scale"], fem_data["LR_scale"]))#低解像度結果をもとにした補間により得られる結果
x_interp -= x_ans_0 
geometry.points = x_interp
geometry.save("interp.vtk") #低解像度結果をもとにした補間により得られる結果


"""
GCNによるノード座標位置の予測

今回、エッジe_ij = (i, j)の特徴量としてx_i - x_j \in R3を予測した。
この推論結果を基に各ノードの座標x_iを求める。
以下はアルゴリズム詳細(話を簡単にするために、x座標のみの1次元で説明。つまりこれまでの説明と異なりx_i \in R)
---アルゴリズム---
(1) x_0 = 0に固定
(2) 例えばx_iが{x_p, x_q, x_r, x_s}と隣接している場合、
    x_i - {(x_p + e_ip) + (x_q + e_iq) + (x_r + e_ir) + (x_s + e_is)}/4. = 0
    したがって、
    x_i - (x_p + x_q + x_r + x_s)/4. = (e_ip + e_iq + e_ir + e_is)/4.
    の関係式を仮定する。
(3) (2)を各ノードで用意し、AX=bの連立一次方程式を作る。ここで
    * x = (x_0, ..., x_n)^T
    * Aは(2)の式より定まるn x n係数行列(nはノード数)。ただし(1)の拘束より、A[0,:] = (1, 0..., 0)。
    * bは(2)の式の右辺。ただし(1)の拘束より、b[0] = 0.
(4) e_ijがGCNの推論結果である以上、(3)の連立一次方程式が解を有する可能性は低い。そこでAの疑似逆行列Mを計算し、x' = Mbをノード座標の予測結果とする。
(4)' もしくはAx=bの解をCG法などで計算。
"""
net = Networks.GCN_edge().to("cuda")
net.load_state_dict(torch.load("weight.pth"))
data = dataset.get(0)

edge_index = data.edge_index.cpu().numpy()
pred = net(data).detach().cpu().numpy()*10. #(E, 3) *pri_processでmm -> cmに単位変換していたので、ここで戻している。
node_num = len(data.x) #ノード数
A = np.zeros((3*node_num, 3*node_num)) #係数行列(3次元なので3n x 3n)
b = np.zeros(3*node_num) #ソース項

#####拘束条件
# Ax = bに対し、x = (x_0, y_0, z_0, x_1, y_1, z_1, ..., x_n, y_n, z_n)^T \in R^3nとする。
A[0,0] = 1.; A[1,1] = 1.; A[2,2] = 1.

#####係数の計算
for i in range(1, node_num):
    connected_edge = np.where(edge_index[0] == i)[0]
    neighbors = edge_index[1][connected_edge] #隣接ノードの集合
    num_neigbors = len(neighbors)
    
    A[3*i, 3*i] = 1.; A[3*i+1, 3*i+1] = 1.; A[3*i+2, 3*i+2] = 1.
    for n in neighbors:
        A[3*i, 3*n] = -1./num_neigbors
        A[3*i+1, 3*n+1] = -1./num_neigbors
        A[3*i+2, 3*n+2] = -1./num_neigbors
            
    b[3*i:3*(i+1)] = np.mean(pred[connected_edge], axis = 0)

# A = coo_matrix(A)
# x_pred = cg(A, b, x0 = x_interp.flatten(), maxiter = 1000)[0]
x_pred = np.linalg.lstsq(A, b)[0]
x_pred = x_pred.reshape((-1, 3)) #各ノードの座標位置予測結果

#####x_predをx_ansと位置合わせ
x_pred = pri.shape_registration(x_pred, x_ans)
geometry.points = x_pred
geometry.save("pred.vtk")
