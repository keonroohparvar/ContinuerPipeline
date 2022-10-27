"""
This file serves as a script to ensure that our model performs correctly. To do this, we will
analyze our model's results on a toy dataset of car images provided by PyTorch. 

Implementation taken from: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw

"""

import torch
import torchvision
import matplotlib.pyplot as plt

def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        print("hi")
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])


def main():
    data = torchvision.datasets.StanfordCars(root=".", download=True)
    show_images(data)





if __name__ == '__main__':
    main()


