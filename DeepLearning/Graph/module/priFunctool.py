import numpy as np
import copy

"""
# ============================================= #
点群の位置合わせ。pointsをtargetに合うようにする。
ref : <P. J. Besl and N. D. Mckay. A mrthod for registration of 3-D shapes>
# ============================================= #

Input : 
    points, targets -> <np:float:(N, 3)>
    max_itr -> <int> 300が十分かどうかは未検討。
Output : reg_points -> <np:float:(N, 3)>
"""
def shape_registration(source, target, max_itr = 300):
    #####クォータニオンをロドリゲスの回転行列に変換
    def rot_q(q):
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2.*(q1*q2 - q0*q3), 2.*(q1*q3 + q0*q2)],
            [2.*(q1*q2 + q0*q3), q0**2 + q2**2 - q1**2 - q3**2, 2.*(q2*q3 - q0*q1)],
            [2.*(q1*q3 - q0*q2), 2.*(q2*q3 + q0*q1), q0**2 + q3**2 - q1**2 - q2**2]
        ])
    num = len(source) #点群数
    mean_t = np.mean(target, axis = 0) #ターゲット点群の重心座標

    point = copy.deepcopy(source)
    for _ in range(max_itr):
        mean_p = np.mean(point, axis = 0)

        S = (point.T)@target / num - mean_p.reshape((3,1))@mean_t.reshape((1, 3)) #参考資料の式(24)
        A = S - S.T
        d = np.array([A[1,2], A[2,0], A[0,1]])
        Q = np.zeros((4,4)) #参考資料の式(25)
        Q[0,0] = np.trace(S)
        Q[0,1:] = d
        Q[1:,0] = d
        Q[1:,1:] = S + S.T - np.trace(S)*np.eye(3)

        lam, v = np.linalg.eig(Q)
        rot = rot_q(v[:,np.argmax(lam)])
        mv = mean_t - rot@mean_p #式(26)

        Trans = np.eye(4)
        Trans[:3,:3] = rot
        Trans[:3,3] = mv

        point = (Trans@(np.concatenate((point, np.ones((num, 1))), axis = 1).T)).T
        point = point[:,:-1]
    
    return point


"""
# ============================================= #
隣接行列の計算
# ============================================= #
Input : cells_dict -> <pv.UnstructuredGrid>
Output: <np:int:(2, E)> COO形式。Eはエッジ数(ノードi <-> j間のエッジを2本として計算)。

Note
* 無向グラフを仮定。
* セルフループなし。
* エッジの重み情報なし(A_{ij} \in {0, 1})。
"""
def get_AdjacencyMatrix(cells_dict):
    node_start = []
    node_end = []

    for cell_type, node_index in cells_dict.items():
        if cell_type == 9:
            for cell in node_index:
                node_start.append(cell)
                node_end.append(np.roll(cell, -1))

                node_start.append(np.roll(cell, -1))
                node_end.append(cell)
        else:
            #####vtkセルタイプ毎に追記していく。現状、4節点矩形メッシュしか扱っていないので、cell_type=9しか実装していない。
            raise NotImplementedError
    
    node_start = np.concatenate(node_start)
    node_end = np.concatenate(node_end)
    edge_index = np.stack((node_start, node_end), axis = 0)
    edge_index = np.unique(edge_index, axis = 1)

    return edge_index

"""
内積のエッジ特徴量を計算
"""
def get_inner_prod_feature(dx, edge_index):
    #####ノルム計算
    h_ij = np.linalg.norm(dx, axis = 1)**2

    #####隣接エッジとの内積の計算
    h_ai = []
    h_ja = []
    for k in range(edge_index.shape[1]):
        i, j = edge_index[:,k] #ノードインデックス番号
        dx_ai = dx[(edge_index[1] == i)*(edge_index[0] != j)] #エッジe_ijに対し、e_aiに関するdxを抽出。ただしe_jiは省く。
        h_ai.append(
            np.mean(
                np.sum(dx_ai*dx[k], axis = 1) #内積計算バッチ処理
                ))
        
        dx_ja = dx[(edge_index[0] == j)*(edge_index[1] != i)]
        h_ja.append(
            np.mean(
                np.sum(dx_ja*dx[k], axis = 1)
            ))
    h_ai = np.array(h_ai)
    h_ja = np.array(h_ja)
    h = np.stack((h_ij, h_ai, h_ja), axis = 1)
    return h