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


parser =  argparse.ArgumentParser()
parser.add_argument('song_dir', type=str, help='The link to the youtube playlist we will pull songs from.')
parser.add_argument('output_dir', type=str, help='Path to the output folder')

args = parser.parse_args()

def convert_mp4_to_wav(mp4_dir, wav_dir):
    pass

def convert_wav_to_spectrogram(wav_dir, out_dir):
    pass

def zscore_normalize_songs(mp4_dir):
    pass

def preprocess_data(mp4_dir, normalize_data, out_dir):
    # Make temp dir for wav files
    wav_dir = f'{mp4_dir}/wav_files'
    os.makedirs(wav_dir, exist_ok=True)

    # Convert all mp4 files to wav files
    convert_mp4_to_wav(mp4_dir, wav_dir)

    # Z-Score Normalize the .wav files
    if normalize_data:
        zscore_normalize_songs(wav_dir)
    
    # Convert the wav forms to spectrograms
    convert_wav_to_spectrogram(wav_dir, out_dir)


if __name__ == '__main__':
    preprocess_data(args.song_dir, False, args.output_dir)