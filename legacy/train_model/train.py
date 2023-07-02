"""
This script will have our model train on .wav files.

Author: Keon Roohparvar
Date: November 30, 2022
"""

# Python Imports
import sys
import os
from datetime import datetime
import time

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

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

def get_data(image_dir, batch_size, number_of_convolutions, device):
    data = AudioDataset(
            root_dir=image_dir,
            song_offset=10,
            song_duration=5,
            transform=None, # SEE IF WE NEED TO CHANGE THIS
            number_of_convolutions= number_of_convolutions,
            device=device
        )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader, data._get_spec_shape()

def save_model(model, model_dir, model_name):
    this_model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), this_model_path)
    print(f'Saved model -> {this_model_path}')


def save_information(train_dir, num_epochs, batch_size, loss_type, learning_rate):
    str_to_save = []
    str_to_save.append(f'Number of Epochs - {num_epochs}')
    str_to_save.append(f'Batch Size - {batch_size}')
    str_to_save.append(f'Loss Type - {loss_type}')
    str_to_save.append(f'Learning Rate - {learning_rate}')

    with open(os.path.join(train_dir, 'information.txt'), 'w') as f:
        f.write("\n".join(str_to_save))



def train_model(train_dir, data, model, loss_type, epochs, batch_size, learning_rate, spectrogram_shape, device):
    # Set parameters for training
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = LossFunction(loss_type)
    noise_schedule = BetaScheduler(T=300, device=device)

    # Save infomration about training to this run's directory
    save_information(train_dir, epochs, batch_size, loss_type, learning_rate)

    # Determine how many times to save results
    NUM_CHECKPOINTS = 5
    epochs_per_checkpoint = int(epochs / NUM_CHECKPOINTS) 

    for epoch in range(epochs):
        # Iterate through data each epoch
        for step, batch in enumerate(data):
            optimizer.zero_grad()

            t = torch.randint(0, noise_schedule.T, (batch_size,), device=device).long()
            x_noisy, noise = noise_schedule.forward_diffusion_sample(batch, t, device=device)
            noise_pred = model(x_noisy.to(device), t)

            # # Prints for debugging!
            # print(f'batch shape: {batch.shape}')
            # print(f'x noisy shape: {x_noisy.shape}')
            # print(f'noise_pred shape: {noise_pred.shape}')
            # print(f'noise shape: {noise.shape}')

            # Saves example spectrogram to temp/ folder
            if epoch == 0:
                if not os.path.isdir(os.path.join(train_dir, 'spectrograms')):
                    os.makedirs(os.path.join(train_dir, 'spectrograms'))
                for idx, item in enumerate(batch):
                    torchvision.utils.save_image(item, os.path.join(train_dir, 'spectrograms', f'spec_step{step}_{idx}.jpg') )
                    

            loss = loss_func.get_loss(noise.to(device), noise_pred)
            loss.backward()
            optimizer.step()

            # if epoch % 20 == 0 and step == 0: 
            if epoch % epochs_per_checkpoint == 0 and step == 0 and epoch != 0:
                print('Saving wav and images...')
                noise_schedule.save_wav(model, train_dir, epoch, spectrogram_shape, device)

        # Print results every few Epochs
        if epoch % 5 == 0:
            # print(f'Epoch {epoch}...')
            print(f"Epoch {epoch} | Loss: {loss.item()} ")

    save_model(model, train_dir, model_name='trained_model')

def main():
    # Set Training Hyperparameters
    LOSS_TYPE = 'l1'
    NUM_EPOCHS = 1000
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4

    # Set training parameters
    DATA_DIR = '../data' if os.path.isdir('../data') else 'data'
    TRAINING_FOLDER_LOCATION = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'runs', datetime.now().strftime('%m-%d_%H_%M_%S')])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'Training location - {TRAINING_FOLDER_LOCATION}')
    if not os.path.isdir(TRAINING_FOLDER_LOCATION):
        os.mkdir(TRAINING_FOLDER_LOCATION)

    # Get Model
    model = SimpleUnet()

    # Get Data
    dataloader, spectrogram_shape = get_data(DATA_DIR, BATCH_SIZE, model.num_convolutions, device)

    # Train model
    train_model(TRAINING_FOLDER_LOCATION, dataloader, model, LOSS_TYPE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, spectrogram_shape, device)

if __name__ == '__main__': 
    main()
