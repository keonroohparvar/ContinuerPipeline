"""
This has the custom implementation for the dataloader required for the continuer.

Using Dataset here: https://huggingface.co/datasets/teticio/audio-diffusion-instrumental-hiphop-256
"""
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.audio_diffusion import Mel
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm


    
class NextImgDataset(Dataset):
    def __init__(self, hub_dataset, transform=None):
        time_start = time.time()
        self.transform = transform

        self.dataset = hub_dataset
        self.song_names = hub_dataset['audio_file']
        self.slices = hub_dataset['slice']


        # print('\n'.join([str(i) + ',' + str(j) for i,j in enumerate(self.song_names)][:100]))
        
        self.slice_count_dict = self._create_slice_dict()
        end = time.time()

        if transform is not None:
            self.dataset.set_transform(transform)
        # print(f'Diff: {end - time_start}')
    
    def _create_slice_dict(self):
        slice_count_dict = {}
        song_names, counts = np.unique(np.array(self.song_names), return_counts=True)

        # print('\n'.join([f'{i} - {j}' for i,j in zip(song_names, counts)]))

        for i in range(len(song_names)):
            #NOTE: It is num_slices - 1 because we are not allowing to take the last slice of each song
            slice_count_dict[song_names[i]] = counts[i] - 1 
        return slice_count_dict

    def __len__(self):
        return np.sum([(i - 1) for i in self.slice_count_dict.values()])
    
    def __getitem__(self, index):
        DEBUG = False
        if DEBUG:
            print(f'\n\nin get item! index: {index}')

        # Initalize values for custom loop to get desired images
        p = 0
        valid_imgs_index = index
        num_slices_in_song = self.slice_count_dict[self.song_names[0]]

        while num_slices_in_song <= valid_imgs_index:
            if DEBUG:
                print(f'\tIn redaction loop...')
                print(f'\tPointer is at Song {self.song_names[p]} slice {self.slices[p]}')
                print(f'\tNum slices in song: {num_slices_in_song}')
                print(f'\tValid imgs index: {valid_imgs_index}')
            p += num_slices_in_song + 1
            valid_imgs_index -= num_slices_in_song
            num_slices_in_song = self.slice_count_dict[self.song_names[p]]
        
        # Once loop breaks, our desired slices are in current song, so we'll get indicies of the
        # original slice, the input slice, and the slice following the input slice
        original_slice_ind = p
        input_slice_ind = p + valid_imgs_index
        output_slice_ind = p + valid_imgs_index + 1

        if DEBUG:
            print('\n------')
            print(f'Original Slice: {self.song_names[original_slice_ind]} | Slice {self.slices[original_slice_ind]}')
            print(f'Input Slice: {self.song_names[input_slice_ind]} | Slice {self.slices[input_slice_ind]}')
            print(f'Original Slice: {self.song_names[output_slice_ind]} | Slice {self.slices[output_slice_ind]}')
            print('-------')

        input_imgs = self.dataset[[original_slice_ind, input_slice_ind, output_slice_ind]]['input']
        # output_img = self.dataset[[output_slice_ind]]['input'][0]
        return torch.cat(input_imgs) 


if __name__ == '__main__':
    dataset_path = 'teticio/audio-diffusion-instrumental-hiphop-256'

    dataset = load_dataset(
        dataset_path,
        split="train",
    )

    print(type(dataset))
    print(dataset)


    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])

    vae = None
    vqvae = None

    def transforms(examples):
        # if args.vae is not None and vqvae.config["in_channels"] == 3:
        # if vae is not None and vqvae == 3:
        if False:
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
        else:
            images = [augmentations(image) for image in examples["image"]]
        return {"input": images}

    dataset_custom = NextImgDataset(
        dataset,
        transform=transforms)

    train_dataloader = torch.utils.data.DataLoader(
            dataset_custom, batch_size=2, shuffle=True)

    for step, (inputs, targets) in enumerate(train_dataloader):
        print(inputs.shape)
        print(targets.shape)
        exit()
        pass