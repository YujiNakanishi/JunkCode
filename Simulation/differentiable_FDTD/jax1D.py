import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import sys
import pandas as pd

dx = 0.01 #格子刻み幅 [m]
N = 1001 #格子数
cfl = 1.
rho = 1.293
k = 1.4e+5
c = jnp.sqrt(k/rho)
dt = cfl*dx/c #時間刻み [s]
time_step = 50
iteration = 10000
lam = 0.01
sigma2 = 1e-3
x = (jnp.arange(N-1)+0.5)*dx
p_init = jnp.exp(-((x-5.)**2)/sigma2) #初期圧力
u_init = jnp.zeros(N) #速度

p_pred_init = jnp.zeros(N-1) #予測
sensor_pos = [470, 490, 510, 530]

@jit
def get_answer(p, u):
    alpha = dt/dx/rho
    beta = dt*k/dx
    p_ans = [jnp.array([p[s] for s in sensor_pos])]
    for itr in range(time_step):
        u = u.at[1:-1].set(u[1:-1]-alpha*(p[1:] - p[:-1]))
        p -= beta*(u[1:] - u[:-1])
        p_ans.append(jnp.array([p[s] for s in sensor_pos]))
    
    p_ans = jnp.stack(p_ans, axis = 0)
    
    return p_ans

p_ans = get_answer(p_init, u_init) #<jnp, float32, (time_step+1, 2)>

@jit
def get_loss(p_pred, p_ans):
    return jnp.mean((p_pred-p_ans)**2)

@jit
def get_grad(p_pred_init, p_ans):
    p_pred = get_answer(p_pred_init, u_init)
    return get_loss(p_pred, p_ans)
get_dp = grad(get_grad)

for itr in range(iteration):
    print(itr)
    p_pred = get_answer(p_pred_init, u_init)
    gradient = get_dp(p_pred_init, p_ans)

    p_pred_init -= lam*gradient

data = [x, np.array(p_init), np.array(p_pred_init)]
data = np.stack(data, axis = 1)
data = pd.DataFrame(data)
data.to_csv("test.csv")