import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import sys
import pandas as pd
import math

def writeVTK(filename, shape, size, scalars = [], scalarname = []):
	if len(shape) == 2:
		shape = (shape[0], shape[1], 1)
		size = (size[0], size[1], 1.)
		scalars = [np.expand_dims(s, axis = -1) for s in scalars]

	with open(filename, "w") as file:
		#####ジオメトリ構造の書き込み
		file.write("# vtk DataFile Version 2.0\nnumpyVTK\nASCII\n")
		file.write("DATASET STRUCTURED_GRID\n")
		file.write("DIMENSIONS "+str(shape[0])+" "+str(shape[1])+" "+str(shape[2])+"\n")
		file.write("POINTS "+str(shape[0]*shape[1]*shape[2])+" float\n")

		for k in range(shape[2]):
			for j in range(shape[1]):
				for i in range(shape[0]):
					file.write(str(i*size[0])+" "+str(j*size[1])+" "+str(k*size[2])+"\n")

		#####スカラーの書き込み
		if scalars != []:
			file.write("POINT_DATA "+str(shape[0]*shape[1]*shape[2])+"\n")
			
			for _scalar, name in zip(scalars, scalarname):
				#####微小量の丸め込み
				scalar = _scalar.copy()
				scalar[np.abs(scalar) < 1e-20] = 0.
				
				file.write("SCALARS "+name+" float\n")
				file.write("LOOKUP_TABLE default\n")

				for k in range(shape[2]):
					for j in range(shape[1]):
						for i in range(shape[0]):
							file.write(str(scalar[i,j,k])+"\n")

dx = 0.01 #格子刻み幅 [m]
N = 101
cfl = 0.5
rho = 1.293
k = 1.4e+5
c = jnp.sqrt(k/rho)
dt = cfl*dx/c #時間刻み [s]
time_step = 50
iteration = 10000
lam = 0.01
sigma2 = 1e-3

u_init = jnp.zeros((N, N-1)) #速度
v_init = jnp.zeros((N-1, N))

p_init = np.zeros((N-1, N-1))
for i in range(N-1):
    for j in range(N-1):
        p_init[i, j] = np.exp(-((i*dx-0.5)**2+(j*dx-0.5)**2)/sigma2) + np.exp(-((i*dx-0.2)**2+(j*dx-0.3)**2)/(sigma2/2.)) + np.exp(-((i*dx-0.6)**2+(j*dx-0.3)**2)/(sigma2/2.))

p_init = jnp.array(p_init)
p_pred_init = jnp.zeros((N-1, N-1)) #予測
sensor_pos = []
for i in range(1,10):
	for j in range(1,10):
		sensor_pos.append((i*10, j*10))

#@jit
def get_answer(p, u, v):
	alpha = dt/dx/rho
	beta = dt*k/dx
	p_ans = [jnp.array([p[s] for s in sensor_pos])]
	for itr in range(time_step):
		writeVTK("result"+str(itr).zfill(3)+".vtk", p.shape, (dx, dx), scalars = [p], scalarname = ["ans"])
		u = u.at[1:-1,:].add(-alpha*(p[1:,:] - p[:-1,:]))
		v = v.at[:,1:-1].add(-alpha*(p[:,1:] - p[:,:-1]))
		p -= beta*(u[1:,:] - u[:-1,:] + v[:,1:] - v[:,:-1])
		p_ans.append(jnp.array([p[s] for s in sensor_pos]))
	sys.exit()
	p_ans = jnp.stack(p_ans, axis = 0)
	return p_ans

p_ans = get_answer(p_init, u_init, v_init) #<jnp, float32, (time_step+1, 10)>


@jit
def get_loss(p_pred, p_ans):
    return jnp.mean((p_pred-p_ans)**2)

@jit
def get_grad(p_pred_init, p_ans):
    p_pred = get_answer(p_pred_init, u_init, v_init)
    return get_loss(p_pred, p_ans)
get_dp = grad(get_grad)

for itr in range(iteration):
    p_pred = get_answer(p_pred_init, u_init, v_init)
    gradient = get_dp(p_pred_init, p_ans)

    p_pred_init -= lam*gradient

    print(str(itr)+"\t"+str(jnp.mean(p_pred_init)))

writeVTK("result.vtk", p_init.shape, (dx, dx), scalars = [p_init, p_pred_init], scalarname = ["ans", "pred"])