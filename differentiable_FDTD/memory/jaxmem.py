import jax
import jax.numpy as jnp
from jax import grad, jit
import time
import jax.profiler

"""
N = 100000000 #格子数
time_step = 10

dx = 0.01 #格子刻み幅 [m]
cfl = 1.
rho = 1.293
k = 1.4e+5
c = jnp.sqrt(k/rho)
dt = cfl*dx/c #時間刻み [s]
sigma2 = 1e-3
x = (jnp.arange(N-1)+0.5)*dx
p_init = jnp.exp(-((x-N*dx*0.5)**2)/sigma2) #初期圧力
u_init = jnp.zeros(N) #速度

p_pred_init = jnp.zeros(N-1) #予測
alpha = dt/dx/rho
beta = dt*k/dx

def step(p, u):
    u = u.at[1:-1].set(u[1:-1]-alpha*(p[1:] - p[:-1]))
    p -= beta*(u[1:] - u[:-1])

    return p, u

def get_answer(p, u):
    for itr in range(time_step):
        p, u = step(p, u)
    return jnp.sum(p**2)

p_ans = get_answer(p_init, u_init) #<jnp, float32, (time_step+1, 2)>

def get_loss(p_pred_init, p_ans):
    p_pred = get_answer(p_pred_init, u_init)
    return jnp.mean((p_pred-p_ans)**2)

get_dp = grad(get_loss)

gradient = get_dp(p_pred_init, p_ans)
gradient.block_until_ready()
"""

x = jnp.zeros(1)
x = 2.*x

jax.profiler.save_device_memory_profile("memory.prof")