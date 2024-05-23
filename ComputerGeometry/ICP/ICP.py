import numpy as np
import pyvista as pv
import copy

#####ロドリゲスの回転行列
def rot(theta, vec):
    nx, ny, nz = vec[0], vec[1], vec[2]
    sin, cos = np.sin(theta), np.cos(theta)

    R = np.array([
        [nx**2*(1.-cos) + cos, nx*ny*(1.-cos) - nz*sin, nx*nz*(1.-cos) + ny*sin],
        [nx*ny*(1.-cos) + nz*sin, ny**2*(1.-cos) + cos, ny*nz*(1.-cos) - nx*sin],
        [nx*nz*(1.-cos) - ny*sin, ny*nz*(1.-cos) + nx*sin, nz**2*(1.-cos) + cos]
    ])

    return R

#####ロドリゲスの回転行列(クォータニオンから変換)
def rot_q(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2.*(q1*q2 - q0*q3), 2.*(q1*q3 + q0*q2)],
        [2.*(q1*q2 + q0*q3), q0**2 + q2**2 - q1**2 - q3**2, 2.*(q2*q3 - q0*q1)],
        [2.*(q1*q3 - q0*q2), 2.*(q2*q3 + q0*q1), q0**2 + q3**2 - q1**2 - q2**2]
    ])

#####Point-to-Pointによる点群位置合わせ
def Point2Point(source, target, max_itr):
    num = len(source) #点群数
    mean_t = np.mean(target, axis = 0) #ターゲット点群の重心座標

    point = copy.deepcopy(source)
    for _ in range(max_itr):
        mean_p = np.mean(point, axis = 0)

        S = (point.T)@target / num - mean_p.reshape((3,1))@mean_t.reshape((1, 3))
        A = S - S.T
        d = np.array([A[1,2], A[2,0], A[0,1]])
        Q = np.zeros((4,4))
        Q[0,0] = np.trace(S)
        Q[0,1:] = d
        Q[1:,0] = d
        Q[1:,1:] = S + S.T - np.trace(S)*np.eye(3)

        lam, v = np.linalg.eig(Q)
        rot = rot_q(v[:,np.argmax(lam)])
        mv = mean_t - rot@mean_p

        Trans = np.eye(4)
        Trans[:3,:3] = rot
        Trans[:3,3] = mv

        point = (Trans@(np.concatenate((point, np.ones((num, 1))), axis = 1).T)).T
        point = point[:,:-1]
    
    return point


if __name__ == "__main__":
    target = np.random.rand(100, 3) #ターゲット点群
    target_geo = pv.UnstructuredGrid()
    target_geo.points = target
    target_geo.save("target.vtk")

    theta = np.pi/4.
    vec = np.ones(3)
    vec = vec/np.linalg.norm(vec) #回転軸

    source = (rot(theta, vec)@target.T).T + np.array([1., 0., 0.])
    source_geo = pv.UnstructuredGrid()
    source_geo.points = source
    source_geo.save("source.vtk")

    point = Point2Point(source, target, 100)
    point_geo = pv.UnstructuredGrid()
    point_geo.points = point
    point_geo.save("point.vtk")