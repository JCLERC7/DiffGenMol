# DDPM functions
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# Timesteps
timesteps = 300

# Beta scheduler
beta_min = 1e-4
beta_max = 2e-2
#TODO: Try others repartitions
betas = torch.linspace(beta_min, beta_max, steps=timesteps)

# Precalculate values
alphas = 1 - betas
alphas_sqrt = torch.sqrt(alphas)
alphas_bar = torch.cumprod(alphas,0)
alphas_bar_sqrt = torch.sqrt(alphas_bar)

# FORWARD PROCESS

# Add noise directly from x_0 to x_t
def add_noise_directly(x_init,t) :
    mean = alphas_bar_sqrt[t] * x_init
    std = torch.sqrt(1-alphas_bar[t])
    noise = torch.randn_like(x_init)
    x_t = mean + std*noise
    return x_t, noise

# REVERSE PROCESS

# Reverse from x_t to x_t-1 with x_0
def reverse_step(x_init, x, t) :
    mean = alphas_bar_sqrt[t-1]*betas[t]/(1-alphas_bar[t])*x_init + alphas_sqrt[t]*(1-alphas_bar[t-1])/(1-alphas_bar[t])*x
    std = torch.sqrt((1-alphas_bar[t-1])/((1-alphas_bar[t]))*betas[t])
    return mean + std*torch.randn_like(x_init)

# LOST FUNCTION

# Calculate loss
def get_loss(model, x_0, t):
    x_t, noise = add_noise_directly(x_0, t)
    noise_pred = model(x_t, t)
    return F.l1_loss(noise, noise_pred)