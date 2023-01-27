"""
This contains the code for a simple block in the U-Net architecture.
"""

from torch import nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, time_embedding_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_embedding_dim, out_channel)

        if up:
            self.conv1 = nn.Conv1d(2*in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_channel, out_channel, 4, 2, padding=1)

        else:
            self.conv1 = nn.Conv1d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv1d(out_channel, out_channel, 4, 2, padding=1)

        self.conv2 = nn.Conv1d(out_channel, out_channel, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_channel)
        self.bnorm2 = nn.BatchNorm1d(out_channel)
        self.relu  = nn.ReLU()
    
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, )]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)