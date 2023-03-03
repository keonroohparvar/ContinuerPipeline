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
from data.util.SpectrogramConverter import SpectrogramConverter
from data.util.SpectrogramParams import SpectrogramParams

params = {
    
    'n_fft': 1024,
    'sr': 44100
}

path_to_wav = '/home/keonroohparvar/2022-2023/winter/csc597/JazzBot/data/jazz_waveforms/jazz.00000.wav'

# waveform, sr = torchaudio.load(
#     path_to_wav,
#     num_frames = params['sr'] * 10)
waveform = pydub.AudioSegment.from_file(path_to_wav, format='wav', duration=20)

params = SpectrogramParams(sample_rate=22050)
sc = SpectrogramConverter(params=params)

spec = sc.spectrogram_from_audio(waveform)

print(spec)
print(type(spec))
print(spec.dtype)
print(spec.shape)

audio_reconstructed = sc.audio_from_spectrogram(spec)
print(audio_reconstructed)

spec_reduced = spec[:, :, :1024]
print(spec_reduced.shape)

audio_reduced_reconstructed = sc.audio_from_spectrogram(spec_reduced)

out_path = 'test/'
# audio_reconstructed.export(out_f=os.path.join(out_path, 'original_reconstructed.wav'), format='wav')
# audio_reduced_reconstructed.export(out_f=os.path.join(out_path, 'reduced_reconstructed.wav'), format='wav')