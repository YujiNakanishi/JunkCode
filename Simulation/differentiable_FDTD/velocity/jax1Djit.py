import jax
import jax.numpy as jnp
from jax import jit
import time

dx = 0.01 #格子刻み幅 [m]
N = 100000000 #格子数
cfl = 1.
rho = 1.293
k = 1.4e+5
c = jnp.sqrt(k/rho)
dt = cfl*dx/c #時間刻み [s]
time_step = 1000
alpha = dt/dx/rho
beta = dt*k/dx

sigma2 = 1e-3
x = (jnp.arange(N-1)+0.5)*dx
p_init = jnp.exp(-((x-N*dx*0.5)**2)/sigma2) #初期圧力
u_init = jnp.zeros(N) #速度

@jit
def step(p, u):
    u = u.at[1:-1].add(-alpha*(p[1:] - p[:-1]))
    p -= beta*(u[1:] - u[:-1])
    return p, u

p, u = step(p_init, u_init)
start = time.time()

for i in range(time_step):
    p, u = step(p, u)

print(time.time()-start)