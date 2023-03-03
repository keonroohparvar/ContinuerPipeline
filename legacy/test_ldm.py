from typing import Optional, Tuple, Union, List
import os

import numpy as np
import torch

from diffusers import LDMPipeline
# from diffusers.models import AutoencoderKL, UNet2DModel
from diffusers.models import AutoencoderKL, UNet2DModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from PIL import Image
import PIL
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# # Local imports
# from data.util.SpectrogramConverter import SpectrogramConverter
# from data.util.SpectrogramParams import SpectrogramParams

class LDMPipeJazz(LDMPipeline):
    def __init__(
        self, 
        vqvae: AutoencoderKL,
        unet: UNet2DModel, 
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        model_path: Optional[str] = None
    ):
        super().__init__()
        self.register_modules(
            vqvae=vqvae,
            unet=unet,
            scheduler=scheduler
        )

        self.model_path = model_path


if __name__ == '__main__':
    model_path = '../../keons_model'
    # sd_pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    # LDMPipeline
    vqvae = AutoencoderKL.from_pretrained(model_path, subfolder='vqvae')
    unet = UNet2DModel.from_pretrained(model_path, subfolder='unet')
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
    # pipe = LDMPipeline()