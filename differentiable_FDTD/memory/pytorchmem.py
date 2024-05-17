import torch
from torch.profiler import profile, record_function, ProfilerActivity

N = 10000 #格子数
time_step = 100

with profile(activities = [ProfilerActivity.CUDA], profile_memory = True, record_shapes = True) as prof:
    dx = 0.01 #格子刻み幅 [m]
    cfl = 1.
    rho = 1.293
    k = 1.4e+5
    c = torch.tensor(k/rho).to("cuda")
    dt = cfl*dx/c #時間刻み [s]
    alpha = dt/dx/rho
    beta = dt*k/dx

    sigma2 = 1e-3
    x = ((torch.arange(N-1)+0.5)*dx).to("cuda")
    p_init = torch.exp(-((x-N*dx*0.5)**2)/sigma2) #初期圧力
    u_init = torch.zeros(N).to("cuda") #速度

    u_index = torch.arange(1,N-1).to("cuda")
    p_index = torch.arange(N-1).to("cuda")
    def step(p, u):
        u = torch.index_add(u, 0, u_index, -alpha*(p[1:] - p[:-1]))
        p = torch.index_add(p, 0, p_index, -beta*(u[1:] - u[:-1]))
        return p, u

    p, u = step(p_init, u_init)
    p.requires_grad = True
    u.requires_grad = True

    
    for i in range(time_step):
        p, u = step(p, u)
    loss = torch.sum(p**2)
    loss.backward()

with open("./pytorchmem.txt", "w") as f:
    f.write(prof.key_averages().table())