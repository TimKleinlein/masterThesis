import pickle
import random

with open('identified_roles.pkl', 'rb') as file:
    roles = pickle.load(file)


random.seed(2)
check_manually = []
for i in range(10):
    sessions = len(roles.keys())
    rand_session = list(roles.keys())[random.randint(0, sessions - 1)]
    dic = roles[rand_session]

    lobbies = len(dic.keys())
    rand_lobby = list(dic.keys())[random.randint(0, lobbies - 1)]
    dic = dic[rand_lobby]

    streamer = len(dic.keys())
    rand_streamer = list(dic.keys())[random.randint(0, streamer - 1)]

    check_manually.append(f'{rand_session}-{rand_lobby}-{rand_streamer}')


print(check_manually)
"""
['2022-01-27_S1-16-2022-01-27_S1_aribunnie_1277953775', '2022-02-15_S1-5-2022-02-15_S1_zeroyalviking_1299154832', 
'2022-05-24_S1-1-2022-05-24_S1_courtilly_1492684956', '2022-02-09_S1-11-2022-02-09_S1_skadj_1292740157', 
'2022-01-27_S1-5-2022-01-27_S1_zeroyalviking_1277951635', '2022-03-01_S1-1-2022-03-01_S1_zeroyalviking_1412348124', 
'2022-03-09_S1-17-2022-03-09_S1_jvckk_1420577772', '2022-02-23_S1-3-2022-02-23_S1_skadj_1307878042', 
'2022-01-27_S1-2-2022-01-27_S1_courtilly_1277953044', '2022-02-19_S1-19-2022-02-19_S1_karacorvus_1303411073']

"""
