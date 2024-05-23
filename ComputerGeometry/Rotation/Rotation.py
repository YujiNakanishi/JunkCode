import numpy as np
import pyvista as pv


def rot(theta, vec):
    nx, ny, nz = vec[0], vec[1], vec[2]
    sin, cos = np.sin(theta), np.cos(theta)

    R = np.array([
        [nx**2*(1.-cos) + cos, nx*ny*(1.-cos) - nz*sin, nx*nz*(1.-cos) + ny*sin],
        [nx*ny*(1.-cos) + nz*sin, ny**2*(1.-cos) + cos, ny*nz*(1.-cos) - nx*sin],
        [nx*nz*(1.-cos) - ny*sin, ny*nz*(1.-cos) + nx*sin, nz**2*(1.-cos) + cos]
    ])

    return R



if __name__ == "__main__":
    point = np.random.rand(100, 3) #ターゲット点群

    theta = np.pi/4. #回転角
    vec = np.ones(3)
    vec = vec/np.linalg.norm(vec) #回転軸

    point2 = (rot(theta, vec)@point.T).T

    geo1 = pv.UnstructuredGrid()
    geo1.points = point
    geo1.save("geo1.vtk")
    geo2 = pv.UnstructuredGrid()
    geo2.points = point2
    geo2.save("geo2.vtk")
