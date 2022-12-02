"""
Author: Keon Roohparvar
Date: Oct 16., 2022

This script serves as all of the preprocessing needed to convert the raw mp4 files into scaled 
spectograms that we can use in our model.
"""

import argparse
import shutil
import os
import subprocess
import json

import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import skimage
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

parser =  argparse.ArgumentParser()
parser.add_argument('song_dir', type=str, help='This is the path to the directory that contains all the downloaded mp4 songs.')
parser.add_argument('output_dir', type=str, help='Path to the output folder')

args = parser.parse_args()

def convert_mp4_to_wav(mp4_dir, wav_dir):
    # Get all files in mp4_dir
    all_mp4_files = [i for i in os.listdir(mp4_dir) if i[-3:] == 'mp4']

    for i in all_mp4_files:
        this_file_location = f'{mp4_dir}/{i}'
        this_file_location = os.path.join(mp4_dir, i)
        this_file_out_location = os.path.join(wav_dir, i[:-3] + 'wav')

        print(f'Converting file at ->\n\t{this_file_location} to ->\n\t{this_file_out_location}')

        # Call subprocess to convert it to wav
        subprocess.call([
            'ffmpeg',
            '-i', this_file_location,
            '-ac', '1',
            '-f', 'wav',
            this_file_out_location
        ])

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def create_single_spectrogram(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)

def convert_wav_to_spectrogram(wav_dir, out_dir):
    # SPECTROGRAM HYPERPARAMETERS
    hop_length = 512
    n_mels = 128
    time_steps = 384


    # Get all files in wav_dir
    all_wav_files = [i for i in os.listdir(wav_dir) if i[-3:] == 'wav']

    samples_dict = {}

    for wav_file in all_wav_files:
        wav_file_location = os.path.join(wav_dir, wav_file)
        print(f'Location: {wav_file_location}')
        this_y, this_sr = librosa.load(wav_file_location, offset=1.0, duration=10, sr=41000)

        # Extract window
        start_sample = 0 # starting at beginning
        length_samples = time_steps*hop_length
        window = this_y[start_sample:start_sample+length_samples]

        this_out_path = os.path.join(out_dir, wav_file[:-3] + 'png')
        create_single_spectrogram(window, this_sr, this_out_path, hop_length, n_mels)

        # Save samples to sample_dict
        samples_dict[wav_file[:-4]] = this_sr

    # Save samples dict out
    sr_out_filepath = os.path.join(out_dir, 'samples_dict.json')
    with open(sr_out_filepath, 'w') as f:
        json.dump(samples_dict, f)

def convert_wav_to_spectrogram2(wav_dir, out_dir):
    # Get all files in wav_dir
    all_wav_files = [i for i in os.listdir(wav_dir) if i[-3:] == 'wav']

    samples_dict = {}

    for wav_file in all_wav_files:
        wav_file_location = os.path.join(wav_dir, wav_file)
        print(f'Location: {wav_file_location}')
        sample_rate, samples = wavfile.read(wav_file_location)
        # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate) 
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(samples, Fs=sample_rate)

        samples_dict[wav_file] = sample_rate

        # Remove borders
        plt.axis('off')
        
        out_location = os.path.join(out_dir, wav_file[:-3] + 'png')
        plt.savefig(out_location, dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True)

    # Save samples dict out
    out_filepath = os.path.join(out_dir, 'samples_dict.json')
    with open(out_filepath, 'w') as f:
        json.dump(samples_dict, f)


def save_samples_dict(wav_dir, out_dir):
    all_wav_files = [i for i in os.listdir(wav_dir) if i[-3:] == 'wav']

    samples_dict = {}

    for wav_file in all_wav_files:
        wav_file_location = os.path.join(wav_dir, wav_file)
        print(f'Location: {wav_file_location}')
        sample_rate, samples = wavfile.read(wav_file_location)

        samples_dict[wav_file] = sample_rate

    # Save dict out
    out_filepath = os.path.join(out_dir, 'samples_dict.json')
    with open(out_filepath, 'w') as f:
        json.dump(samples_dict, f)


def convert_spectrogram_to_wav(spec_dir, out_dir):
    # Get all files in spec_dir
    all_spec_files = [i for i in os.listdir(spec_dir) if i[-3:] == 'png']

    # HARD CODING PARAMS
    hop_length = 512
    n_fft = hop_length * 2
    sr = 41000


    # read in dict with all sample rates
    sample_rate_dict_location = os.path.join(spec_dir, '..', 'samples_dict.json')
    with open(sample_rate_dict_location, 'r') as f:
        sample_rate_dict = json.load(f)
    
    print(sample_rate_dict)

    for spec_file in all_spec_files:
        this_sample_rate = sample_rate_dict[spec_file[:-4]]
        print(f'This sample rate: {this_sample_rate}')

        # this_spec = plt.imread(os.path.join(spec_dir, spec_file), format='png')
        this_spec = skimage.io.imread(os.path.join(spec_dir, spec_file), format='png')

        print(this_spec.shape)
        print(this_spec)


        # _, this_wav = signal.istft(this_spec, fs=this_sample_rate)
        this_wav = librosa.feature.inverse.mel_to_audio(this_spec, sr=sr, n_fft=n_fft, hop_length=hop_length)
        # print(this_wav)
        # return

        # save wavfile out
        this_wav_filename = os.path.join(out_dir, spec_file[:-3] + '_reconstructed.wav')
        wavfile.write(this_wav_filename, this_sample_rate, this_wav)

        return



def preprocess_data(mp4_dir, normalize_data, out_dir):
    # Make temp dir for wav files
    wav_dir = f'{mp4_dir}/wav_files'
    wav_dir = os.path.join(mp4_dir, 'wav_files')
    os.makedirs(wav_dir, exist_ok=True)

    # Convert all mp4 files to wav files
    convert_mp4_to_wav(mp4_dir, wav_dir)
    
    # Convert the wav forms to spectrograms
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    convert_wav_to_spectrogram(wav_dir, out_dir)


if __name__ == '__main__':
    # song_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/data_raw'
    # out_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_specs'
    # preprocess_data(song_dir, False, out_dir)

    spec_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_specs'
    out_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_wav_dir'
    convert_spectrogram_to_wav(spec_dir, out_dir)
    
    
    
    # preprocess_data(args.song_dir, False, args.output_dir)
    # save_samples_dict(f'{args.song_dir}/wav_files', args.output_dir)
    # convert_spectrogram_to_wav('/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/jazz', '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_wav_dir')
    
