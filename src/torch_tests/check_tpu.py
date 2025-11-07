import torch

device = torch.device("xla")
print("Using device:", device)

x = torch.randn(3, 3, device=device)
y = torch.randn(3, 3, device=device)
z = x + y

print("Operation result:", z)
print("On device:", z.device)
