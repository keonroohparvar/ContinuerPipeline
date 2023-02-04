"""
This script will test if we can do the spectrogram stuff!
"""

import torchaudio
import sys
import os
import pydub

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

path_to_wav = '/home/keonroohparvar/2022-2023/winter/csc597/JazzBot/data/Breezin.wav'

# waveform, sr = torchaudio.load(
#     path_to_wav,
#     num_frames = params['sr'] * 10)
waveform = pydub.AudioSegment.from_file(path_to_wav, format='wav', duration=20)


print(waveform)
print(type(waveform))

params = SpectrogramParams()
sc = SpectrogramConverter(params=params)

spec = sc.spectrogram_from_audio(waveform)

print(spec)
print(spec.shape)

audio_reconstructed = sc.audio_from_spectrogram(spec)
print(audio_reconstructed)

out_path = '/home/keonroohparvar/2022-2023/winter/csc597/JazzBot/train_model/test/test.wav'
audio_reconstructed.export(out_f=out_path, format='wav')