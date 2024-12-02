import pickle as pkl
import random
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

with open('data/final_transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

with open('../../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobbies = pkl.load(file)

def extract_segment(input_file, output_file, start_time, end_time):
    # start_time and end_time are in seconds
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

random.seed(2)

for acc in ['low_acc', 'mid_acc', 'high_acc']:
    i = 0
    while i < 10:
        disc_rounds = len(transcriptions.keys())
        rand_disc_round = list(transcriptions.keys())[random.randint(0, disc_rounds - 1)]
        dic = transcriptions[rand_disc_round]

        utterances = len(dic[acc])
        if utterances == 0:
            continue
        rand_utt = dic[acc][random.randint(0, utterances - 1)]
        i += 1

        print(f'{rand_disc_round} - {acc}: {rand_utt}')

        ses = rand_disc_round.split('_')[0] + '_' + rand_disc_round.split('_')[1]
        lob = int(rand_disc_round.split('_')[2][1:])
        streamer_short = rand_disc_round.split('_')[4]
        for s in lobbies[ses][lob].keys():
            if streamer_short in s:
                streamer = s

        # extract relevant lobby
        lobby_start = lobbies[ses][lob][streamer][0].total_seconds()
        lobby_end = lobbies[ses][lob][streamer][1].total_seconds()

        extract_segment(f'../../../pop520978/data/{ses}/{streamer}.mkv',
                        f'evaluation/videos/{rand_disc_round}.mkv', lobby_start,
                        lobby_end)


