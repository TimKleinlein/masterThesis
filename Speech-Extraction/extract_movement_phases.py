import pickle
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import torch
import wave
import contextlib

with open('output_discussion_rounds/final_discussion_rounds.pkl', 'rb') as file:
    discussion_rounds = pickle.load(file)

with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobbies = pickle.load(file)

# function to extract segments from videos
def extract_segment(input_file, output_file, start_time, end_time):
    # start_time and end_time are in seconds
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

# convert to audio wav file
def convert_mkv_to_wav(mkv_file, wav_file):
    video = VideoFileClip(mkv_file)
    audio = video.audio
    audio.write_audiofile(wav_file)

def extract_segment_audio(input_filename, output_filename, start_time, end_time):
    with wave.open(input_filename, 'rb') as infile:
        # Get the properties needed to configure the output file
        n_channels = infile.getnchannels()
        sample_width = infile.getsampwidth()
        framerate = infile.getframerate()

        # Set positions in terms of frames
        start_frame = int(start_time * framerate)
        end_frame = int(end_time * framerate)
        duration_frames = end_frame - start_frame

        # Set the read position
        infile.setpos(start_frame)

        # Read and extract the desired frames
        frames = infile.readframes(duration_frames)

        # Create the output file
        with wave.open(output_filename, 'wb') as outfile:
            outfile.setnchannels(n_channels)
            outfile.setsampwidth(sample_width)
            outfile.setframerate(framerate)
            outfile.writeframes(frames)

# create dictionary of relevant movement phases
relevant_movement_phases = {}

for ses in discussion_rounds.keys():
    relevant_movement_phases[ses] = {}
    for lob in discussion_rounds[ses].keys():
        relevant_movement_phases[ses][lob] = []

        lobby_length = int((lobbies[ses][lob][list(lobbies[ses][lob].keys())[0]][1] - lobbies[ses][lob][list(lobbies[ses][lob].keys())[0]][0]).total_seconds())

        def get_movement_phases(discussion_phases):
            if discussion_phases:  # not empty list
                # Start from 0 up to start of first discussion round
                movement_phases = [[0, discussion_phases[0][0]]]

                # Iterate through the other discussion phases to find remaining movement phases
                for i in range(len(discussion_phases) - 1):
                    start = discussion_phases[i][1]
                    end = discussion_phases[i + 1][0]
                    movement_phases.append([start, end])

                movement_phases.append([discussion_phases[-1][1], lobby_length])

                return movement_phases
            else:
                return []


        # get movement phases
        test_dr = discussion_rounds[ses][lob]
        movement_phases = get_movement_phases(test_dr)

        # only use first two as option
        first_movement_phases = movement_phases[:2]

        # only use the ones with a minimum length of 1 minute
        final_movement_phases = [x for x in first_movement_phases if (x[1] - x[0]) > 60]

        for sl in final_movement_phases:
            relevant_movement_phases[ses][lob].append(sl)


# find all participating streamers
streamers = []
for s in lobbies.keys():
    for l in lobbies[s].keys():
        for str in lobbies[s][l].keys():
            name = str[14:]
            name = name.split('_')[0]
            streamers.append(name)

unique_streamers = list(set(streamers))

"""
# create directories
for s in unique_streamers:
    os.mkdir(f'train_data_diarization/{s}')

for s in unique_streamers:
    for ses in relevant_movement_phases.keys():
        for lob in relevant_movement_phases[ses].keys():
            if relevant_movement_phases[ses][lob]:
                streamers_of_lobby_original_names = list(lobbies[ses][lob].keys())
                streamers_of_lobby = [x[14:].split('_')[0] for x in streamers_of_lobby_original_names]
                if s in streamers_of_lobby:
                    original_streamer_name = streamers_of_lobby_original_names[streamers_of_lobby.index(s)]
                    for ind, mp in enumerate(relevant_movement_phases[ses][lob]):
                        streamer_movement_start = lobbies[ses][lob][original_streamer_name][0].total_seconds() + mp[0]
                        streamer_movement_end = lobbies[ses][lob][original_streamer_name][0].total_seconds() + mp[1]
                        if streamer_movement_start < 0:  # because adding silence doesnt help when looking for streamers utterances
                            continue
                        extract_segment(f'../../../../dimstore/pop520978/data/{ses}/{original_streamer_name}.mkv',
                                        f'train_data_diarization/{s}/{ses}_{lob}_{ind}.mkv', streamer_movement_start, streamer_movement_end)
                        convert_mkv_to_wav(f'train_data_diarization/{s}/{ses}_{lob}_{ind}.mkv', f'train_data_diarization/{s}/{ses}_{lob}_{ind}.wav')
                        if os.path.exists(f'train_data_diarization/{s}/{ses}_{lob}_{ind}.mkv'):
                            os.remove(f'train_data_diarization/{s}/{ses}_{lob}_{ind}.mkv')

"""

# run pyannote over extracted audios to find suitable phrases
auth_token = "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"
auth_token_write = "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"
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',
                                    use_auth_token='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')
pipeline = pipeline.to(torch.device('cuda:0'))

unique_streamers_first_gpu = ['falcone', 'aribunnie', 'skadj', 'ayanehylo', 'brizzynewindsong', 'chilledchaos', 'pjonk', 'jayfletcher88', 'aplatypuss', 'uneasypeasy', 'heckmuffins', 'sidearms4reason', 'kaywordley', 'cheesybluenips', 'hcjustin', 'br00d', 'junkyard129', 'ozzaworld', 'taydertot', 'x33n', 'itsdanpizza', 'vikramafc']


for s in unique_streamers_first_gpu:
    files = os.listdir(f'train_data_diarization/{s}')
    os.mkdir(f'train_data_diarization/{s}/utterances')
    for mp in files:
        with contextlib.closing(wave.open(f'train_data_diarization/{s}/{mp}', 'rb')) as file:
            frames = file.getnframes()
            rate = file.getframerate()
            duration = frames / float(rate)

        diarization = pipeline(f'train_data_diarization/{s}/{mp}')
        # first check if audio has multiple speakers in interval start + 15 and end - 15 seconds and if so exclude it
        main_speaker_set = False
        multiple_speakers = False
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start < 15 or turn.end > (duration - 15):
                continue
            if not main_speaker_set:
                main_speaker = speaker
                main_speaker_set = True
            if speaker != main_speaker:
                multiple_speakers = True
                break
        if multiple_speakers:
            continue
        # now continue by extracting all the utterances of the main speaker (short ones will be removed later on)
        utterance_number = 1
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start < 15 or turn.end > (duration - 15):
                continue
            extract_segment_audio(f'train_data_diarization/{s}/{mp}', f'train_data_diarization/{s}/utterances/{mp[:-4]}_{utterance_number}.wav', turn.start, turn.end)
            utterance_number += 1

