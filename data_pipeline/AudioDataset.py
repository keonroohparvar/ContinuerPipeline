"""
This file will be the script that loads in data from a folder into pytorch tensors.
"""

from torch.utils.data import Dataset
import torchaudio
import os
import pydub

# Local Imports
from .SpectrogramConverter import SpectrogramConverter
from .SpectrogramParams import SpectrogramParams


class AudioDataset(Dataset):
    def __init__(self, root_dir, song_offset, song_duration, sample_rate=44100, transform=None, device='cuda'):
        self.root_dir = root_dir
        self.song_duration = song_duration
        self.song_offset = song_offset

        self.num_frames_of_waveform = self._format_len_of_song(sample_rate)

        # Instantiate object to help convert between spectrograms <-> audio
        self.spec_converter = SpectrogramConverter(SpectrogramParams(), device=device)
        
        # Save all song paths in dir
        self.song_paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i[-3:] == 'wav']
        self.sample_rates = {}

        self.transform = transform
         
    def _format_len_of_song(self, sample_rate):
        """
        This function is used to make sure the length of the song is a power of 2
        """
        NUMBER_OF_CONVOLUTIONS = 6
        num_frames = sample_rate * self.song_duration
        num_frames_div_by = 2**NUMBER_OF_CONVOLUTIONS
        num_frames_of_waveform = num_frames - (num_frames % num_frames_div_by)
        
        return num_frames_of_waveform

    def _convert_wav_to_spectrogram(self, x):
        """
        This uses the SpectrogramConverter class to convert our waveforms to Spectrograms
        which we will be using for our training.
        """
        return self.spec_converter.spectrogram_from_audio(x)

    def _input_size_calculation(self, length, num_convolutions):
        """
        This shortens the length of the data to allow it to be convoluded down without issues in
        our UNet Model.
        """
        return length - (length % (2 ** num_convolutions))

    def _convert_spec_to_waveform(self, spec):
        """
        This uses the SpectrogramConverter class to convert our Spectrograms to waveforms
        which we will be using for our training.
        """
        return self.spec_converter.audio_from_spectrogram(spec, apply_filters=False) # TODO: See if this changes anything

    def __len__(self):
        return len(self.song_paths)
    
    
    def __getitem__(self, index):
        # Get path
        song_path = self.song_paths[index]

        # Get Metadata to find out how much to load
        metadata = torchaudio.info(song_path)

        waveform = pydub.AudioSegment.from_file(
            song_path, 
            format='wav', 
            duration=self.song_duration)

        spec = self._convert_wav_to_spectrogram(waveform)

        spec = spec[:, :, :self._input_size_calculation(spec.shape[2], num_convolutions=6)]

        return spec
