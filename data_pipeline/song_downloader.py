"""
Author: Keon Roohparvar
Date: October 17, 2022

This script downloads mp3 files from a jazz playlist. 

Chosen Playlist = https://www.youtube.com/playlist?list=PLiy0XOfUv4hFHmPs0a8RqkDzfT-2nw7WV
"""

import argparse
import shutil
import os

import numpy as np
from pytube import Playlist

parser =  argparse.ArgumentParser()
parser.add_argument('link_to_playlist', type=str, help='The link to the youtube playlist we will pull songs from.')
parser.add_argument('output_dir', type=str, help='Path to the output folder')
parser.add_argument('--num_songs', type=int, help='The number of songs we want to download. Leave empty if we want to download the entire playlist.')

args = parser.parse_args()


def download_songs(link_to_playlist, output_dir, num_songs):
    p = Playlist(link_to_playlist)

    videos = p.videos

    if num_songs:
        chosen_videos = np.random.choice(videos, num_songs, replace=False)
    
    else:
        chosen_videos = videos

    for video in chosen_videos:
        print(f'Downloading title: {video.title}')
        title_to_save = (video.title).replace(' ', '').lower()

        this_audio = video.streams.filter(only_audio=True).first()
        this_audio.download(output_dir + '/' + title_to_save)

    print('Done downloading songs.')

def format_output_dir(output_dir):
    print('Formatting output directory...')

    dirs = os.listdir(output_dir)

    for dir in dirs:
        file_to_move = os.listdir(f'{output_dir}/{dir}')[0]
        shutil.move(f'{output_dir}/{dir}/{file_to_move}', f'{output_dir}/{file_to_move}')
        print('removing ', f'{output_dir}/{dir}/')
        os.removedirs(f'{output_dir}/{dir}/')
    
    print('Done with formatting output directory.')

if __name__ == '__main__':
    link = 'https://www.youtube.com/playlist?list=PLiy0XOfUv4hFHmPs0a8RqkDzfT-2nw7WV'
    out_dir = '/home/keonroohparvar/2022-2023/fall/csc596/JazzBot/data_raw'
    num_songs = 10
    download_songs(link, out_dir, num_songs)
    format_output_dir(out_dir)