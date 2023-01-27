"""
This file will be the script that loads in dummy data to confirm the model can learn something.
"""
import torch
from torch.utils.data import Dataset
import torchaudio
import os


class DummyAudioDataset(Dataset):
    def __init__(self, root_dir, song_offset, song_duration, transform=None):
        self.root_dir = root_dir
        self.song_duration = song_duration
        self.song_offset = song_offset

        # Save all song paths in dir
        self.song_paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i[-3:] == 'wav']
        self.sample_rates = {}

        self.transform = transform
    

    def _format_len_of_song(self, sample_rate):
        """
        This function is used to make sure the length of the song is a power of 2
        """
        num_frames = sample_rate * self.song_duration
        num_frame_pow_2 = 2
        while (num_frame_pow_2 ** 2) < num_frames:
            num_frame_pow_2 = num_frame_pow_2 ** 2
        
        return num_frame_pow_2

       

    def __len__(self):
        return len(self.song_paths)
    
    def __getitem__(self, index):
        # Get path
        song = self.song_paths[index]

        # Get Metadata to find out how much to load
        metadata = torchaudio.info(song)
        sample_rate = metadata.sample_rate
        song_duration_pow_2 = self._format_len_of_song(sample_rate)
        # print(f'song dur pow 2: {song_duration_pow_2}')
        waveform, _ = torchaudio.load(song, 
                                    frame_offset=sample_rate*self.song_offset, 
                                    num_frames=song_duration_pow_2)
        
        dummy_data = torch.zeros_like(waveform)

        # print(f'dummy data shape: {dummy_data.shape}')
        
        return dummy_data
