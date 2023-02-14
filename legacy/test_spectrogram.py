"""
This script will test if we can do the spectrogram stuff!
"""

import torchaudio
import sys
import os
import pydub
import numpy as np

# Add parent dir to path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

print(parent_path)

# Local imports
from data_pipeline.SpectrogramConverter import SpectrogramConverter
from data_pipeline.SpectrogramParams import SpectrogramParams

params = {
    
    'n_fft': 1024,
    'sr': 44100
}

path_to_wav = '../data/Breezin.wav'

# waveform, sr = torchaudio.load(
#     path_to_wav,
#     num_frames = params['sr'] * 10)
waveform = pydub.AudioSegment.from_file(path_to_wav, format='wav', duration=20)

params = SpectrogramParams()
sc = SpectrogramConverter(params=params)

spec = sc.spectrogram_from_audio(waveform)

print(spec)
print(type(spec))
print(spec.shape)

audio_reconstructed = sc.audio_from_spectrogram(spec)
print(audio_reconstructed)

spec_reduced = spec[:, :, :1024]
print(spec_reduced.shape)

audio_reduced_reconstructed = sc.audio_from_spectrogram(spec_reduced)

out_path = 'test/'
# audio_reconstructed.export(out_f=os.path.join(out_path, 'original_reconstructed.wav'), format='wav')
# audio_reduced_reconstructed.export(out_f=os.path.join(out_path, 'reduced_reconstructed.wav'), format='wav')