"""
This file serves as a script to ensure that our model performs correctly. To do this, we will
analyze our model's results on a toy dataset of car images provided by PyTorch. 

Initial implementation taken from
    -> https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw

"""
# Global Imports
import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from diffusion_model.LossFunction import LossFunction
from diffusion_model.noise_scheduler import BetaScheduler
from diffusion_model.time_embedding import SinusoidalPositionEmbeddings
from diffusion_model.model_architecture import SimpleUnet



def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])


def train_model(data, model, loss_type, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_func = LossFunction(loss_type)
    noise = BetaScheduler(T=300)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, noise.T, (batch_size,), device=device).long()
            x_noisy, noise = noise.forward_diffusion_sample(batch[0], t, device)
            noise_pred = model(x_noisy, t)


            loss = loss_func.get_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                noise.sample_plot_image()



def main():
    # Get Data
    data = torchvision.datasets.StanfordCars(root=".", download=True)
    show_images(data)

    # Get Model
    model = None

    # Train model
    train_model(data, model)





if __name__ == '__main__':
    # main()
    print('hi')


