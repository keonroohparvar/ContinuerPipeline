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
    def __init__(self, root_dir, song_offset, song_duration, number_of_convolutions=6, sample_rate=44100, transform=None, device='cuda'):
        self.root_dir = root_dir
        self.song_duration = song_duration
        self.song_offset = song_offset

        # Save number of convolutions which will determine the size of our spectrograms
        self.num_convolutions = number_of_convolutions

        # Instantiate object to help convert between spectrograms <-> audio
        self.spec_converter = SpectrogramConverter(SpectrogramParams(), device=device)
        
        # Save all song paths in dir
        self.song_paths = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i[-3:] == 'wav']
        self.sample_rates = {}

        self.transform = transform

    def _get_spec_shape(self):
        """
        This will get the shape of a spectrogram which we will ultimately need when 
        creating new spectrograms.
        """
        example_wav = pydub.AudioSegment.from_file(
            self.song_paths[0], 
            format='wav', 
            duration=self.song_duration)

        spec =  self._convert_wav_to_spectrogram(example_wav)
        spec = spec[:, :, :self._input_size_calculation(spec.shape[2], self.num_convolutions)]
        return spec.shape

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

        spec = spec[:, :, :self._input_size_calculation(spec.shape[2], self.num_convolutions)]

        return spec
