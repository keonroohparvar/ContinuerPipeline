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

# import numpy as np
# import librosa
# from scipy.io import wavfile

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


# def preprocess_data(mp4_dir, normalize_data, out_dir):
#     # Make temp dir for wav files
#     wav_dir = f'{mp4_dir}/wav_files'
#     wav_dir = os.path.join(mp4_dir, 'wav_files')
#     os.makedirs(wav_dir, exist_ok=True)

#     # Convert all mp4 files to wav files
#     convert_mp4_to_wav(mp4_dir, wav_dir)
    
#     # Convert the wav forms to spectrograms
#     if not os.path.isdir(out_dir):
#         os.makedirs(out_dir)
#     convert_wav_to_spectrogram(wav_dir, out_dir)


if __name__ == '__main__':
    convert_mp4_to_wav('/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/data_raw', '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/data')
    # song_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/data_raw'
    # out_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_specs'
    # preprocess_data(song_dir, False, out_dir)

    # spec_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_specs'
    # out_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_wav_dir'
    # convert_spectrogram_to_wav(spec_dir, out_dir)
    
    
    
    # preprocess_data(args.song_dir, False, args.output_dir)
    # save_samples_dict(f'{args.song_dir}/wav_files', args.output_dir)
    # convert_spectrogram_to_wav('/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/jazz', '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/spectrograms/new_wav_dir')
    
