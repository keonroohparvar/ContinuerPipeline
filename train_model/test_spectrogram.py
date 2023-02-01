# I cant believe I am doing this :(

from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Load in wav
path_to_wav = '/Users/keonroohparvar/Documents/School/2022-2023/Winter/csc597/JazzBot/data/Breezin.wav'
samples, sample_rate = librosa.load(path_to_wav, sr=None)
# sample_rate, samples = wavfile.read(path_to_wav)

print(samples)
print(min(samples))
print(max(samples))


plt.figure(figsize=(14, 5))
librosa.display.waveshow(samples, sr=sample_rate)

sgram = librosa.stft(samples)
# librosa.display.specshow(sgram)

sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

# print('mel_scale_sgram')
# print(mel_scale_sgram)

# mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

# librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')

# print(mel_sgram)

# # plt.show()

# Reverse process
sgram_mag_inverse = librosa.feature.inverse.mel_to_audio(mel_scale_sgram, sr=sample_rate)
print('inverse!')
print(sgram_mag_inverse)
print('original')
print(mel_scale_sgram)




