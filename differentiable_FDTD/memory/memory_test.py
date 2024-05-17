import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities = [ProfilerActivity.CUDA], profile_memory = True, record_shapes = True) as prof:
    x = torch.zeros(100).to("cuda")
    x.requires_grad = True
    y = torch.zeros(100).to("cuda")
    y.requires_grad = True
    loss = torch.mean((x-y)**2)
    loss.backward()

with open("./pytorchmem.txt", "w") as f:
    f.write(prof.key_averages().table())