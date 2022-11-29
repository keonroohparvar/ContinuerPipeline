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

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
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

def convert_wav_to_spectrogram(wav_dir, out_dir):
    # Get all files in wav_dir
    all_wav_files = [i for i in os.listdir(wav_dir) if i[-3:] == 'wav']

    for wav_file in all_wav_files:
        wav_file_location = os.path.join(wav_dir, wav_file)
        print(f'Location: {wav_file_location}')
        sample_rate, samples = wavfile.read(wav_file_location)
        # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate) 
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(samples, Fs=sample_rate)

        plt.axis('off')

        # # Remove borders
        plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False) 

        
        out_location = os.path.join(out_dir, wav_file[:-3] + 'png')
        # plt.savefig(out_location, dpi=300, frameon='false')
        plt.savefig(out_location, dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True)
        # plt.savefig(out_location, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.imsave(out_location, )

def zscore_normalize_songs(mp4_dir):
    pass

def preprocess_data(mp4_dir, normalize_data, out_dir):
    # Make temp dir for wav files
    wav_dir = f'{mp4_dir}/wav_files'
    wav_dir = os.path.join(mp4_dir, 'wav_files')
    os.makedirs(wav_dir, exist_ok=True)

    # Convert all mp4 files to wav files
    # convert_mp4_to_wav(mp4_dir, wav_dir)

    # Z-Score Normalize the .wav files
    if normalize_data:
        zscore_normalize_songs(wav_dir)
    
    # Convert the wav forms to spectrograms
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    convert_wav_to_spectrogram(wav_dir, out_dir)


if __name__ == '__main__':
    preprocess_data(args.song_dir, False, args.output_dir)
    