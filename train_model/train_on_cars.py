"""
This file serves as a script to ensure that our model performs correctly. To do this, we will
analyze our model's results on a toy dataset of car images provided by PyTorch. 

Initial implementation taken from
    -> https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw

"""
# Global Imports
import sys
import os
from datetime import datetime

import torch
import torchvision
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Add parent dir to path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

# Local imports
from diffusion_model.LossFunction import LossFunction
from diffusion_model.noise_scheduler import BetaScheduler
from diffusion_model.time_embedding import SinusoidalPositionEmbeddings
from diffusion_model.model_architecture import SimpleUnet

def get_car_data(img_size, batch_size):

    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=os.path.dirname(os.path.abspath(__file__)), download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=os.path.dirname(os.path.abspath(__file__)), download=True, 
                                         transform=data_transform, split='test')
    data = torch.utils.data.ConcatDataset([train, test])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])

def save_img_to_image_dir(save_dir, img):
    pass


def train_model(train_dir, data, model, loss_type, epochs, batch_size):
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

            t = torch.randint(0, noise_schedule.T, (batch_size,), device=device).long()
            x_noisy, noise = noise_schedule.forward_diffusion_sample(batch[0], t, device)
            noise_pred = model(x_noisy, t)


            loss = loss_func.get_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                # noise_schedule.sample_plot_image(64, device, model)
                noise_schedule.save_img_to_image_dir(train_dir, epoch, 64, device, model)

def main():
    # Set training parameters
    TRAINING_FOLDER_LOCATION = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'runs', datetime.now().strftime('%m-%d_%H_%M_%S')])
    IMG_SIZE = 64

    print(TRAINING_FOLDER_LOCATION)

    if not os.path.isdir(TRAINING_FOLDER_LOCATION):
        os.mkdir(TRAINING_FOLDER_LOCATION)
    
    # Set Hyperparameters
    LOSS_TYPE = 'l1'
    NUM_EPOCHS = 100
    BATCH_SIZE = 128

    # Get Data
    dataloader = get_car_data(IMG_SIZE, BATCH_SIZE)

    # Get Model
    model = SimpleUnet()

    # Train model
    train_model(TRAINING_FOLDER_LOCATION, dataloader, model, LOSS_TYPE, NUM_EPOCHS, BATCH_SIZE)

if __name__ == '__main__':
    main()
    # print('hi')
