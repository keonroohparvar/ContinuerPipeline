"""
Author: Keon Roohparvar
Date: 2/22/2023

This is the implementation of the pipeline for the Jazzbot that confirms to DiffusionPipeline from HuggingFace.

"""

from diffusers import DiffusionPipeline
import torch


class JazzbotPipeline(DiffusionPipeline):
    def __init__(self, unet, vae, eta) -> None:
        super().__init__()

        self.eta = eta

        self.register_modules(unet=unet, vae=vae)


    def __call__(self, batch_size: int = 1, num_inference_steps: int = 50):
        # Sample gaussian noise to begin loop
        image = torch.randn((batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size))

        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, self.eta).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image