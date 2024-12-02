import pandas as pd
import pickle

# TODO:  2022-02-17_S1_L2 and 2022-02-26_S1_L14 have wrong lobby end
me_rounds = pd.read_excel('output_discussion_rounds/ME_DR.xlsx')

with open('output_discussion_rounds/final_discussion_rounds_from_pyannote.pkl', 'rb') as file:
    automated_rounds = pickle.load(file)

# create new dictionary in which automated discussion round results are combined with timestamps from manual extraction
discussion_rounds = automated_rounds.copy()

me_rounds['Number Discussion Rounds'] = (me_rounds.count(axis=1) - 1) / 2
for s in me_rounds['Session']:
    ses = s[:13]
    lob = int(s[15:])
    disc_rounds = []
    for i in range(int(me_rounds[me_rounds['Session'] == s]['Number Discussion Rounds'].values[0])):
        start_time = me_rounds[me_rounds['Session'] == s][f'DR{int(i+1)} S'].values[0]
        end_time = me_rounds[me_rounds['Session'] == s][f'DR{int(i+1)} E'].values[0]
        disc_rounds.append([start_time.hour * 60 + start_time.minute,
                            end_time.hour * 60 + end_time.minute])
    discussion_rounds[ses][lob] = disc_rounds

with open('output_discussion_rounds/final_discussion_rounds.pkl', 'wb') as file:
    pickle.dump(discussion_rounds, file)
