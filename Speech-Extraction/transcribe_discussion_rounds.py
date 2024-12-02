import pickle
import stable_whisper
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import os

# TODO: subclip extraction is not precise to millisecond

# function to extract segments from videos
def extract_segment(input_file, output_file, start_time, end_time):
    # start_time and end_time are in seconds
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

# convert to audio wav file
def convert_mkv_to_wav(mkv_file, wav_file):
    video = VideoFileClip(mkv_file)
    audio = video.audio
    audio.write_audiofile(wav_file)

def extract_middle_part(string):
    parts = string.split('_')
    if len(parts) > 2:
        return parts[2]
    else:
        return None

with open(f'output_discussion_rounds/final_discussion_rounds.pkl',
          'rb') as file:
    discussion_rounds = pickle.load(file)

with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobby_times = pickle.load(file)

relevant_streamers = os.listdir('Personal_VAD_Model/data/LibriSpeech/dev-clean')
if '.DS_Store' in relevant_streamers:
    relevant_streamers.remove('.DS_Store')

# create directory to store transcriptions
#os.mkdir('relevant_transcriptions')
# create directory to save discussion rounds wav files for target speaker extraction
#os.mkdir('relevant_discussion_rounds_audio')

model = stable_whisper.load_model("large", device='cuda:0')
print("Whisper model loaded.")

sessions = [
  '2022-01-26_S1', '2022-01-27_S1', '2022-01-28_S1', '2022-02-01_S1',
  '2022-02-02_S1', '2022-02-04_S1', '2022-02-05_S1', '2022-02-08_S1', '2022-02-09_S1', '2022-02-10_S1',
 '2022-02-12_S1', '2022-02-15_S1', '2022-02-16_S1', '2022-02-17_S1', '2022-02-19_S1',
 '2022-02-21_S1', '2022-02-22_S1', '2022-02-23_S1', '2022-02-24_S1', '2022-02-26_S1', '2022-03-01_S1', '2022-03-02_S1', '2022-03-03_S1',
 '2022-03-09_S1', '2022-03-10_S1', '2022-05-24_S1', '2022-05-24_S2']

for s in sessions:
    for l in list(discussion_rounds[s].keys()):
        for d, i in enumerate(discussion_rounds[s][l]):
            if discussion_rounds[s][l][d][0] - 2 > 0:
                start = discussion_rounds[s][l][d][0] - 2
            else:
                start = discussion_rounds[s][l][d][0]

            lobby_length = (lobby_times[s][l][list(lobby_times[s][l].keys())[0]][1] - lobby_times[s][l][list(lobby_times[s][l].keys())[0]][0]).total_seconds()
            if discussion_rounds[s][l][d][1] + 2 < lobby_length:
                end = discussion_rounds[s][l][d][1] + 2
            else:
                end = discussion_rounds[s][l][d][1]

            # for now transcriptions of all relevant streamers
            streamers = []
            options = list(lobby_times[s][l].keys())
            for opt in options:
                if extract_middle_part(opt) in relevant_streamers:
                    streamers.append(opt)

            for streamer in streamers:
                streamer_discussion_start = lobby_times[s][l][streamer][0].total_seconds() + start
                streamer_discussion_end = lobby_times[s][l][streamer][0].total_seconds() + end

                if streamer_discussion_start < 0:  # in rare cases where streamer did not participate in the first discussion round (2022-02-05_S1 - 1 - 2022-02-05_S1_karacorvus_1288211105 - 0, 2022-02-15_S1 - 14 - 2022-02-15_S1_irepptar_1299294423 - 0, 2022-02-17_S1 - 3 - 2022-02-17_S1_zeroyalviking_1301249902 - 0)
                    continue

                extract_segment(f'../../pop520978/data/{s}/{streamer}.mkv', f'{s}_l{l}_{d}_{streamer}.mkv', streamer_discussion_start,
                                streamer_discussion_end)
                convert_mkv_to_wav(f'{s}_l{l}_{d}_{streamer}.mkv', f'{s}_l{l}_{d}_{streamer}.wav')

                # transcribe and save in srt format
                result = model.transcribe(audio=f'{s}_l{l}_{d}_{streamer}.wav')
                result.to_srt_vtt(f'{s}_l{l}_{d}_{streamer}.srt')

                # rename srt and wav file with streamer name only without session and number and put in directory relevant_transcriptions
                orig_streamer_name = extract_middle_part(streamer)
                os.rename(f'{s}_l{l}_{d}_{streamer}.wav',
                          f'relevant_discussion_rounds_audio/{s}_l{l}_{d}_{orig_streamer_name}.wav')
                os.rename(f'{s}_l{l}_{d}_{streamer}.srt',
                          f'relevant_transcriptions/{s}_l{l}_{d}_{orig_streamer_name}.srt')

                # remove mkv if not needed anymore, keep wav file as input for personal vad speaker diarization model
                if os.path.exists(f'{s}_l{l}_{d}_{streamer}.mkv'):
                    os.remove(f'{s}_l{l}_{d}_{streamer}.mkv')
                # if os.path.exists(f'{s}_l{l}_{d}_{streamer}.wav'):
                #     os.remove(f'{s}_l{l}_{d}_{streamer}.wav')
