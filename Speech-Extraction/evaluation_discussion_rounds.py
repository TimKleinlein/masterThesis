import pickle
import random

with open('output_discussion_rounds/final_discussion_rounds.pkl', 'rb') as file:
    disc_rounds = pickle.load(file)


with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobby_times = pickle.load(file)

random.seed(2)
check_manually = []
for i in range(10):
    sessions = len(disc_rounds.keys())
    rand_session = list(disc_rounds.keys())[random.randint(0, sessions - 1)]
    dic = disc_rounds[rand_session]

    lobbies = len(dic.keys())
    rand_lobby = list(dic.keys())[random.randint(0, lobbies - 1)]
    dic = dic[rand_lobby]

    discussion_rounds = len(dic)
    rand_discussion_rounds = random.randint(0, discussion_rounds - 1) + 1

    streamer = len(lobby_times[rand_session][rand_lobby].keys())
    rand_streamer = list(lobby_times[rand_session][rand_lobby].keys())[random.randint(0, streamer - 1)]

    start_time = lobby_times[rand_session][rand_lobby][rand_streamer][0].total_seconds() + dic[rand_discussion_rounds - 1][0]
    end_time = lobby_times[rand_session][rand_lobby][rand_streamer][0].total_seconds() + dic[rand_discussion_rounds - 1][1]


    check_manually.append(f'{rand_session}-{rand_lobby}-{rand_discussion_rounds}-{rand_streamer}'
                          f'-{start_time}-{end_time}')


print(check_manually)
"""
['2022-01-27_S1-16-2022-01-27_S1_aribunnie_1277953775', '2022-02-15_S1-5-2022-02-15_S1_zeroyalviking_1299154832', 
'2022-05-24_S1-1-2022-05-24_S1_courtilly_1492684956', '2022-02-09_S1-11-2022-02-09_S1_skadj_1292740157', 
'2022-01-27_S1-5-2022-01-27_S1_zeroyalviking_1277951635', '2022-03-01_S1-1-2022-03-01_S1_zeroyalviking_1412348124', 
'2022-03-09_S1-17-2022-03-09_S1_jvckk_1420577772', '2022-02-23_S1-3-2022-02-23_S1_skadj_1307878042', 
'2022-01-27_S1-2-2022-01-27_S1_courtilly_1277953044', '2022-02-19_S1-19-2022-02-19_S1_karacorvus_1303411073']

"""
