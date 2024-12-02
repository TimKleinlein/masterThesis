from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import pickle
import torch
import numpy as np


# speaker diarization
auth_token = "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"
auth_token_write = "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"


# function to extract segments from videos
def extract_segment(input_file, output_file, start_time, end_time):
    # start_time and end_time are in seconds
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

# convert to audio wav file
def convert_mkv_to_wav(mkv_file, wav_file):
    video = VideoFileClip(mkv_file)
    audio = video.audio
    audio.write_audiofile(wav_file)


# test how good diarization performs
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',
                                    use_auth_token='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')
pipeline = pipeline.to(torch.device('cuda:0'))


# run code for one entire lobby: 2022-02-16_S1 L7
with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobbies = pickle.load(file)


sessions = [
 '2022-01-26_S1', '2022-01-27_S1', '2022-01-28_S1', '2022-02-01_S1', '2022-02-02_S1', '2022-02-04_S1', '2022-02-05_S1',
 '2022-02-08_S1', '2022-02-09_S1', '2022-02-10_S1', '2022-02-12_S1', '2022-02-15_S1', '2022-02-16_S1', '2022-02-17_S1', '2022-02-19_S1',
 '2022-02-21_S1', '2022-02-22_S1', '2022-02-23_S1', '2022-02-24_S1', '2022-02-26_S1', '2022-03-01_S1', '2022-03-02_S1', '2022-03-03_S1',
 '2022-03-09_S1', '2022-03-10_S1', '2022-05-24_S1', '2022-05-24_S2']



for l in sessions:

    pot_dis_starts = {}
    final_dic = {}

    streamers = list(lobbies[l][2].keys())


    for ind, s in enumerate(streamers):
        if ind > 0:
            continue
        start_time = lobbies[l][2][s][0].seconds
        end_time = lobbies[l][2][s][1].seconds
        extract_segment(f'../../../../dimstore/pop520978/data/{l}/{s}.mkv', f'{s}_l2.mkv', start_time, end_time)
        convert_mkv_to_wav(f'{s}_l2.mkv', f'{s}_audio.wav')
