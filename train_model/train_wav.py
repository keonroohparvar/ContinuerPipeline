"""
This script will have our model train on .wav files.

Author: Keon Roohparvar
Date: November 30, 2022
"""

# Python Imports
import sys
import os
from datetime import datetime

import torch
import torchaudio
from torch.optim import Adam
from torch.utils.data import DataLoader

# Add parent dir to path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

# Local imports
from data_pipeline.AudioDataset import AudioDataset
from diffusion_model.LossFunction import LossFunction
from diffusion_model.noise_scheduler import BetaScheduler
from diffusion_model.time_embedding import SinusoidalPositionEmbeddings
from diffusion_model.model_architecture import SimpleUnet

def get_data(image_dir, batch_size):
    # data_transforms = [
    #     transforms.Resize((img_size, img_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(), # Scales data into [0,1] 
    #     transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    # ]
    # data_transform = transforms.Compose(data_transforms)

    data = AudioDataset(
            root_dir=image_dir,
            song_offset=10,
            song_duration=10,
            transform=None # SEE IF WE NEED TO CHANGE THIS
        )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader

def save_model(model, model_dir, epoch_num):
    this_model_path = os.path.join(model_dir, f'model_{str(epoch_num)}')
    torch.save(model.state_dict(), this_model_path)
    print(f'Saved model -> {this_model_path}')


def train_model(train_dir, data, model, loss_type, epochs, batch_size):
    # CONSTANTS WE NEED
    WAV_SIZE = 441000
    SAMPLE_RATE = 44100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_func = LossFunction(loss_type)
    noise_schedule = BetaScheduler(T=300)

    for epoch in range(epochs):
        for step, batch in enumerate(data):
            if epoch % 5 == 0 and step == 0:
                print(f'Epoch {epoch}...')
            
            optimizer.zero_grad()

            # print(batch[0])
            # print(batch[0].shape)
            # print('-----')
            # print(batch[0][0])
            # print(batch[0][0].shape)



            t = torch.randint(0, noise_schedule.T, (batch_size,), device=device).long()
            x_noisy, noise = noise_schedule.forward_diffusion_sample(batch, t, device)
            noise_pred = model(x_noisy, t)

            print(noise_pred)
            print(noise_pred.shape)
            print("------")
            print(noise)
            print(noise.shape)


            loss = loss_func.get_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                # noise_schedule.sample_plot_image(64, device, model)
                
                noise_schedule.save_wav_to_wavs_dir(model, train_dir, epoch, WAV_SIZE, SAMPLE_RATE, device)

def main():
    # Set training parameters
    DATA_DIR = '../data'
    TRAINING_FOLDER_LOCATION = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'runs', datetime.now().strftime('%m-%d_%H_%M_%S')])

    print(TRAINING_FOLDER_LOCATION)

    if not os.path.isdir(TRAINING_FOLDER_LOCATION):
        os.mkdir(TRAINING_FOLDER_LOCATION)
    
    # Set Hyperparameters
    LOSS_TYPE = 'l1'
    NUM_EPOCHS = 100
    BATCH_SIZE = 4

    # Get Data
    dataloader = get_data(DATA_DIR, BATCH_SIZE)

    # Get Model
    model = SimpleUnet()

    # Train model
    train_model(TRAINING_FOLDER_LOCATION, dataloader, model, LOSS_TYPE, NUM_EPOCHS, BATCH_SIZE)



if __name__ == '__main__': 
    main()