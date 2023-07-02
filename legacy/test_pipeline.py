from datasets import load_dataset, load_from_disk
from torchvision.transforms import Compose, Normalize, ToTensor


import numpy as np
import os
import sys
from diffusers.models import AutoencoderKL, UNet2DModel
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusers.pipelines.audio_diffusion import Mel
import argparse
import soundfile as sf
import torch

from continuer_pipeline import ContinuerPipeline

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy.NextImgDataset import NextImgDataset

def main():
    DATA_DIR="teticio/audio-diffusion-instrumental-hiphop-256"
    OUT_DIR = '/Users/keonroohparvar/Documents/School/2022-2023/Winter/csc597/JazzBot/results/train_dataset_imgs'
    dataset = load_dataset(
                    DATA_DIR,
                    use_auth_token=None,
                    split="train"
                )
    dataset = dataset[:50]
    imgs = dataset['image']
    song_names = dataset['audio_file']
    slices = dataset['slice']
    # print(song_names)

    root_img = imgs[0]
    next_img = imgs[1]
    print(type(root_img))
    root_img.save(os.path.join(OUT_DIR, 'root_img.jpg'))
    next_img.save(os.path.join(OUT_DIR, 'next_img.jpg'))
    
                
    resolution = root_img.height, root_img.width

    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])
        
    mel = Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=512,
        sample_rate=22050,
        n_fft=2048
    )

    root_wav = mel.image_to_audio(root_img)
    sf.write(os.path.join(OUT_DIR, 'root_wav.wav'), root_wav, 22050)
    next_wav = mel.image_to_audio(next_img)
    sf.write(os.path.join(OUT_DIR, 'next_wav.wav'), next_wav, 22050)

    model_path = '../keons_continuer'

    unet = UNet2DModel.from_pretrained(model_path, subfolder='unet')
    vqvae = AutoencoderKL.from_pretrained(model_path, subfolder='vqvae')
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
    mel = Mel.from_pretrained(model_path, subfolder='mel')

    pipe = ContinuerPipeline(
        unet = unet,
        scheduler = scheduler,
        vqvae = vqvae,
        mel = mel
    )

    next_img_pred = pipe(
        root_img = root_img,
        out_dir = '/Users/keonroohparvar/Documents/School/2022-2023/Winter/csc597/JazzBot/results/train_dataset_imgs/pipe_output'
    )

    next_img_pred.save(os.path.join(OUT_DIR, 'next_img_prediction.jpg'))
    next_wav_pred = mel.image_to_audio(next_img_pred)
    sf.write(os.path.join(OUT_DIR, 'next_wav_pred.wav'), next_wav_pred, 22050)


    exit()




if __name__ == "__main__":
    main()
