"""
This file will be the script that loads in data from a folder into pytorch tensors.
"""

from torchvision import transforms
import os


class DataLoader:
    def __init__(self, dir, image_shape):
        self.dir = dir
        self.image_shape = image_shape


    def load_in_dataset(self):
        data_transform = transforms.Compose([
        transforms.Resize(self.image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ])

        files = os.listdir(self.dir)

        data = []

        for file in [j for j in files if j[-3:] == 'png']:
            data.append(data_transform(file))

        return data





