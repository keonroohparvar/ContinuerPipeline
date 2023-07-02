"""
This contains the PyTorch implementation of the U-Net model. 

"""
import torch
from torch import nn

from .time_embedding import SinusoidalPositionEmbeddings
from .unet_block import Block
    
class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1

        # Defining Challens
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        # down_channels = (64, 128, 256, 256, 256, 256)
        # up_channels = (256, 256, 256, 256, 128, 64)
        # down_channels = (64, 128, 256)
        # up_channels = (256, 128, 64)

        assert len(down_channels) == len(up_channels)
        self.num_convolutions = len(down_channels)

        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 3, padding=1)
    
    def forward(self, x, timestep):
        # print(f'Initial x shape: {x.shape}')

        # Embed time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # print(f'x shape after: {x.shape}')
        residual_inputs = []

        # Prints for debugging
        # print('before down')
        # print(torch.cuda.memory_allocated())

        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            
        # Prints for debugging
        # print('after down')
        # print(torch.cuda.memory_allocated())

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # print(f'r - {residual_x.shape}')
            # print(f'r - {x.shape}')
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)

        # print(f'x shape after up: {x.shape}')
        return self.output(x)
        