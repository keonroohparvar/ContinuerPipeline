"""
This file will be the script that loads in data from a folder into pytorch tensors.
"""

from torch.utils.data import Dataset
import torchaudio
import os


class AudioDataset(Dataset):
    def __init__(self, root_dir, song_offset, song_duration, transform=None):
        self.root_dir = root_dir
        self.song_duration = song_duration
        self.song_offset = song_offset

        # Save all song paths in dir
        self.song_paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i[-3:] == 'wav']
        self.sample_rates = {}

        self.transform = transform
       

    def __len__(self):
        return len(self.song_paths)
    
    def __getitem__(self, index):
        # Get path
        song = self.song_paths[index]

        # Get Metadata to find out how much to load
        metadata = torchaudio.info(song)
        sample_rate = metadata.sample_rate
        waveform, _ = torchaudio.load(song, 
                                    frame_offset=sample_rate*self.song_offset, 
                                    num_frames=sample_rate*self.song_duration)
                                    
        self.sample_rates[song] = sample_rate

        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform





