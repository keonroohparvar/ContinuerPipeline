"""
This function is the implementation of the class that handles the noise scheduler.
"""

import torch
import torch.nn.functional as F


class BetaScheduler:
    def __init__(self, type='linear'):
        self.type = type

    def get_beta_schedule(self, timesteps, start, end):
        """
        Main function that returns the beta schedule.

        """
        if self.type == 'linear':
            return torch.linspace(start, end, timesteps)
        
        else:
            raise NotImplementedError()

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals while considering the batch
        dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion_sample(self, betas, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        # Terms calculated in closed form
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)