from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import pickle
import torch
import os
import numpy as np
from pydub import AudioSegment


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
    final_lobbies = pickle.load(file)

sessions = ['2022-01-28_S1', '2022-02-24_S1', '2022-02-15_S1']

for ses in sessions:

    if ses == '2022-01-28_S1':
        lobbies = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if ses == '2022-02-24_S1':
        lobbies = [16, 17, 18, 19, 20, 21]
    if ses == '2022-02-15_S1':
        lobbies = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    for l in lobbies:
        pot_dis_starts = {}
        final_dic = {}

        streamers = list(final_lobbies[ses][l].keys())
        if final_lobbies[ses][l][streamers[0]][0] == final_lobbies[ses][l][streamers[0]][1]:  # if lobby has duration = 0
            continue

        for s in streamers:
            if final_lobbies[ses][l][s][0].days < 0:
                start_time = 0
                end_time = final_lobbies[ses][l][s][1].seconds
                missing_seconds = abs(final_lobbies[ses][l][s][0].total_seconds())
                extract_segment(f'../../../../dimstore/pop520978/data/{ses}/{s}.mkv', f'{s}_l{l}.mkv', start_time,
                                end_time)
                convert_mkv_to_wav(f'{s}_l{l}.mkv', f'{s}_l{l}_audio.wav')
                # add silence
                original_audio = AudioSegment.from_wav(f'{s}_l{l}_audio.wav')
                silence = AudioSegment.silent(duration=missing_seconds * 1000)
                final_audio = silence + original_audio
                final_audio.export(f'{s}_l{l}_audio.wav', format="wav")

            else:
                start_time = final_lobbies[ses][l][s][0].seconds
                end_time = final_lobbies[ses][l][s][1].seconds
                extract_segment(f'../../../../dimstore/pop520978/data/{ses}/{s}.mkv', f'{s}_l{l}.mkv', start_time,
                                end_time)
                convert_mkv_to_wav(f'{s}_l{l}.mkv', f'{s}_l{l}_audio.wav')

        for s in streamers:

            diarization = pipeline(f"{s}_l{l}_audio.wav")

            potential_discussion_starts = []
            second_last_speaker = "SPEAKER_01"
            second_last_speaker_start = 0
            last_speaker = "SPEAKER_01"
            last_speaker_start = 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                current_speaker = speaker
                current_speaker_start = turn.start
                if current_speaker == second_last_speaker or current_speaker == last_speaker or last_speaker == second_last_speaker:
                    second_last_speaker = last_speaker
                    second_last_speaker_start = last_speaker_start
                    last_speaker = current_speaker
                    last_speaker_start = current_speaker_start
                    continue
                else:
                    if abs(current_speaker_start - second_last_speaker_start) < 10:
                        potential_discussion_starts.append([second_last_speaker_start, turn.end])
                    second_last_speaker = last_speaker
                    second_last_speaker_start = last_speaker_start
                    last_speaker = current_speaker
                    last_speaker_start = current_speaker_start

            pot_dis_starts[s] = potential_discussion_starts

            # now go through created list of potential discussion starts and extract the timestamps lying close to each other
            extracted_discussion_rounds = {}
            if len(potential_discussion_starts) != 0:
                new_start = potential_discussion_starts[0][0]
            round_n = 1
            for ind, t in enumerate(potential_discussion_starts):
                if ind == 0:
                    continue
                if t[0] - potential_discussion_starts[ind - 1][0] < 30:
                    if t[1] < potential_discussion_starts[ind - 1][1]:
                        potential_discussion_starts[ind][1] = potential_discussion_starts[ind - 1][1]
                    if ind != len(potential_discussion_starts) - 1:
                        continue
                    else:  # last potential discussion start, thus register as last potential discussion in this lobby
                        extracted_discussion_rounds[round_n] = [new_start, t[1]]
                else:
                    extracted_discussion_rounds[round_n] = [new_start, potential_discussion_starts[ind - 1][1]]
                    new_start = t[0]
                    round_n += 1
                    if ind == len(potential_discussion_starts) - 1:
                        extracted_discussion_rounds[round_n] = [new_start, t[1]]

            final_dic[s] = extracted_discussion_rounds
            if os.path.exists(f'{s}_l{l}.mkv'):
                os.remove(f'{s}_l{l}.mkv')
            if os.path.exists(f'{s}_l{l}_audio.wav'):
                os.remove(f'{s}_l{l}_audio.wav')

        with open(f'final_discussion_rounds_s_{ses}_l_{l}.pkl', 'wb') as file:
            pickle.dump(final_dic, file)

