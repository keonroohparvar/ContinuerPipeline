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

# Print cuda availability
if torch.cuda.is_available():
    print(f'Cuda is available!')
    torch.cuda.set_device(0)
else:
    print(f'Cuda is not available :(')

# Add parent dir to path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

# Local imports
from data_pipeline.DummyAudioDataset import DummyAudioDataset
from data_pipeline.AudioDataset import AudioDataset
from diffusion_model.LossFunction import LossFunction
from diffusion_model.noise_scheduler import BetaScheduler
from diffusion_model.time_embedding import SinusoidalPositionEmbeddings
from diffusion_model.model_architecture import SimpleUnet

def get_data(image_dir, batch_size):
    data = AudioDataset(
            root_dir=image_dir,
            song_offset=10,
            song_duration=5,
            transform=None # SEE IF WE NEED TO CHANGE THIS
        )

    size_of_waveforms = data.num_frames_of_waveform

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader, size_of_waveforms

def save_model(model, model_dir, epoch_num):
    this_model_path = os.path.join(model_dir, f'model_{str(epoch_num)}')
    torch.save(model.state_dict(), this_model_path)
    print(f'Saved model -> {this_model_path}')


def train_model(train_dir, data, model, loss_type, epochs, batch_size, wav_size):
    # CONSTANTS WE NEED
    SAMPLE_RATE = 44100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_func = LossFunction(loss_type)
    noise_schedule = BetaScheduler(T=300)

    for epoch in range(epochs):
        if epoch % 5 == 0:
            print(f'Epoch {epoch}...')
        for step, batch in enumerate(data):
            optimizer.zero_grad()

            t = torch.randint(0, noise_schedule.T, (batch_size,), device=device).long()
            x_noisy, noise = noise_schedule.forward_diffusion_sample(batch, t, device)
            noise_pred = model(x_noisy, t)

            loss = loss_func.get_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

            if epoch % 20 == 0 and step == 0 and epoch != 0:
                print('Saving wav...')
                noise_schedule.save_wav_to_wavs_dir(model, train_dir, epoch, wav_size, SAMPLE_RATE, device)

def main():
    # Set training parameters
    DATA_DIR = '../data' if os.path.isdir('../data') else 'data'

    TRAINING_FOLDER_LOCATION = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'runs', datetime.now().strftime('%m-%d_%H_%M_%S')])

    print(f'Training location - {TRAINING_FOLDER_LOCATION}')

    if not os.path.isdir(TRAINING_FOLDER_LOCATION):
        os.mkdir(TRAINING_FOLDER_LOCATION)
    
    # Set Hyperparameters
    LOSS_TYPE = 'l1'
    NUM_EPOCHS = 100
    BATCH_SIZE = 4

    # Get Data
    dataloader, num_frames_in_waveform = get_data(DATA_DIR, BATCH_SIZE)

    # # THE BELOW IS FOR THE DUMMY DATA
    # dummy_data = DummyAudioDataset(
    #         root_dir=DATA_DIR,
    #         song_offset=10,
    #         song_duration=10,
    #         transform=None
    #     )
    # dataloader = DataLoader(dummy_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Get Model
    model = SimpleUnet()

    # Train model
    train_model(TRAINING_FOLDER_LOCATION, dataloader, model, LOSS_TYPE, NUM_EPOCHS, BATCH_SIZE, num_frames_in_waveform)



if __name__ == '__main__': 
    main()
