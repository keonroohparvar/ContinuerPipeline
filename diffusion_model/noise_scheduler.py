"""
This function is the implementation of the class that handles the noise scheduler.
"""

import os

import torch
from torchvision import transforms
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class BetaScheduler:
    def __init__(self, T, type='linear'):
        # Set Type
        self.type = type

        # Get Beta schedule
        self.T = T
        self.betas = self.get_beta_schedule(timesteps=T)

    def get_beta_schedule(self, timesteps, start=0.0001, end=0.02):
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
    
    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        print('x shape')
        print(x_0.shape)
        # Terms calculated in closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        noise = torch.randn_like(x_0)
        self.sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return self.sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    def show_tensor_image(self, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :] 
        plt.imshow(reverse_transforms(image))
    
    @torch.no_grad()
    def sample_timestep(self, x, t, model):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample_plot_image(self, IMG_SIZE, device, model):
        # Sample noise
        img_size = IMG_SIZE
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.T/num_images)

        for i in range(0, self.T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t, model)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize+1))
                self.show_tensor_image(img.detach().cpu())
        plt.show()      
    
    @torch.no_grad()
    def save_wav_to_wavs_dir(self, model, save_dir, epoch_num, WAV_SIZE, sample_rate, device):
        # Create epoch folder
        epoch_folder_name = os.path.join(save_dir, str(epoch_num))
        if not os.path.isdir(epoch_folder_name):
            os.mkdir(epoch_folder_name)

        # Sample noise
        wav = torch.randn((1, 1, WAV_SIZE), device=device)
        NUM_WAVS = 5
        stepsize = int(self.T / NUM_WAVS)

        for i in range(0, self.T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            wav = self.sample_timestep(wav, t, model)
            if i % stepsize == 0:
                # reverse_trCnsforms = transforms.Compose([
                # transforms.Lambda(lambda t: (t + 1) / 2),
                # transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                # transforms.Lambda(lambda t: t * 255.),
                # transforms.Lambda(lambda t: t.cpu()),
                # transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                # transforms.ToPILImage(),
                # ])

                # Take first wav of batch
                if len(wav.shape) == 3:
                    wav_to_save = wav[0, :, :] 
                    # plt.subplot(1, num_imgs, int(i/stepsize+1))
                
                # img_to_save_transformed = reverse_transforms(img_to_save)
                # img_to_save_transformed.save(os.path.join(epoch_folder_name, f'epoch{epoch_num}_step{i}.jpg'))
        
                # Save the wavform
                print('Saving wavform...')
                torchaudio.save(os.path.join(epoch_folder_name, f'epoch{epoch_num}_step{i}.wav'), wav_to_save, sample_rate)