import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg import bicgstab
from module import priFunctool as pri
import autograd
import sys
from scipy import optimize

"""
エッジ特徴量dx_ij = (dx, dy, dz)からノード位置座標を計算する。

input:
    dx -> <np:float:(E, 3)>エッジ特徴量
    x -> <np:float:(N, 3)>ノード位置座標初期候補解
    edge_index -> <np:float:(2, E)> 隣接行列(COO形式)
    sym -> <bool> dx_ij = -dx_jiの後処理を施すか否か
    max_itr -> <int> bicgstab最大反復回数
"""
def get_coord(dx, x, edge_index, sym = True, max_itr = 100):
    E = len(dx) #エッジ数

    if sym:
        for e in range(E):
            i, j = edge_index[0][e], edge_index[1][e]
            if i > j:
                k = np.where((edge_index[0] == j)*(edge_index[1] == i))[0][0]
                dx[e] = -dx[k]
    
    N = len(x) #ノード数
    A = lil_matrix((3*N, 3*N)) #係数行列
    b = np.zeros(3*N)

    #####拘束条件
    A[0,0] = 1.; A[1,1] = 1.; A[2,2] = 1.

    #####係数の計算
    for i in range(1, N):
        connected_edge = np.where(edge_index[0] == i)[0]
        neighbors = edge_index[1][connected_edge] #隣接ノードの集合
        num_neigbors = len(neighbors)
    
        A[3*i, 3*i] = 1.; A[3*i+1, 3*i+1] = 1.; A[3*i+2, 3*i+2] = 1.
        for n in neighbors:
            A[3*i, 3*n] = -1./num_neigbors
            A[3*i+1, 3*n+1] = -1./num_neigbors
            A[3*i+2, 3*n+2] = -1./num_neigbors
            
        b[3*i:3*(i+1)] = np.mean(dx[connected_edge], axis = 0)
    
    x = bicgstab(A, b, x0 = x.flatten(), maxiter = max_itr)[0]
    return x.reshape((-1, 3))




def Newton_Raphson(func_list, x):
    def function(x):
        return np.array([func(x) for func in func_list])
    
    def Jacob(x):
        return np.stack([autograd.grad(func)(x) for func in func_list], axis = 0)
    
    solution = optimize.root(function, x, jac = Jacob)
    return solution.x