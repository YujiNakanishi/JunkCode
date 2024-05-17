import torch
import time

dx = 0.01 #格子刻み幅 [m]
N = 100000000 #格子数
cfl = 1.
rho = 1.293
k = 1.4e+5
c = torch.tensor(k/rho).to("cuda")
dt = cfl*dx/c #時間刻み [s]
time_step = 1000
alpha = dt/dx/rho
beta = dt*k/dx

sigma2 = 1e-3
x = ((torch.arange(N-1)+0.5)*dx).to("cuda")
p_init = torch.exp(-((x-N*dx*0.5)**2)/sigma2) #初期圧力
u_init = torch.zeros(N).to("cuda") #速度

u_index = torch.arange(1,N-1).to("cuda")
def step(p, u):
    u = torch.index_add(u, 0, u_index, -alpha*(p[1:] - p[:-1]))
    p -= beta*(u[1:] - u[:-1])

    return p, u

p, u = step(p_init, u_init)
start = time.time()

for i in range(time_step):
    p, u = step(p, u)

print(time.time()-start)