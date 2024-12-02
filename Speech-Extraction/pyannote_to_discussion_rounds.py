import os
import pickle
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
from datetime import timedelta


sessions = [
 '2022-01-26_S1', '2022-01-27_S1', '2022-01-28_S1', '2022-02-01_S1', '2022-02-02_S1', '2022-02-04_S1', '2022-02-05_S1',
 '2022-02-08_S1', '2022-02-09_S1', '2022-02-10_S1', '2022-02-12_S1', '2022-02-15_S1', '2022-02-16_S1', '2022-02-17_S1', '2022-02-19_S1',
 '2022-02-21_S1', '2022-02-22_S1', '2022-02-23_S1', '2022-02-24_S1', '2022-02-26_S1', '2022-03-01_S1', '2022-03-02_S1', '2022-03-03_S1',
 '2022-03-09_S1', '2022-03-10_S1', '2022-05-24_S1', '2022-05-24_S2']

sessions_discussion_lengths = pd.read_excel('StreamsSpeakingBehavior.xlsx')

with open(f'../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl',
          'rb') as file:
    lobbies = pickle.load(file)

final_discussion_rounds = {}

for s in sessions:
    final_discussion_rounds[s] = {}
    ses_lobbies_all = list(lobbies[s].keys())
    ses_lobbies = []
    for l in ses_lobbies_all:
        if lobbies[s][l] == {}:  # 2022-02-21_S1 L1-6 is empty dictionary somehow
            continue
        if lobbies[s][l][list(lobbies[s][l].keys())[0]][0] != lobbies[s][l][list(lobbies[s][l].keys())[0]][1]:  # if lobby not empty because start == end
            ses_lobbies.append(l)
    for ses_l in ses_lobbies:
        with open(f'output_pyannote/final_discussion_rounds_s_{s}_l_{ses_l}.pkl', 'rb') as file:
            disc_rounds = pickle.load(file)


        new_dic = {}
        for k in disc_rounds.keys():
            new_dic[k] = {}
            already_combined = False
            new_dic_dr_counter = 1
            for l in disc_rounds[k].keys():
                if l == len(disc_rounds[k]):  # last proposed dr
                    if already_combined:
                        continue
                    else:  # it is also a dr proposal in new dic
                        new_dic[k][new_dic_dr_counter] = disc_rounds[k][l]
                else:
                    if already_combined:  # proposed dr has already been merged to previous one
                        already_combined = False
                        continue
                    else:
                        if disc_rounds[k][l][1] - disc_rounds[k][l][0] > 60:  # if proposed dr is longer than 60 seconds it wont be merged
                            new_dic[k][new_dic_dr_counter] = disc_rounds[k][l]
                            new_dic_dr_counter += 1
                        else:  # if shorter than 60 seconds check if following dr is close by and then either merge or not
                            if disc_rounds[k][l+1][0] - disc_rounds[k][l][1] < 60:  # they lie closer than 60 seconds to each other so will be merged
                                new_dic[k][new_dic_dr_counter] = [disc_rounds[k][l][0], disc_rounds[k][l+1][1]]
                                new_dic_dr_counter += 1
                                already_combined = True
                            else:  # if following dr proposal is far away check if previous dr proposal is close by
                                if l == 1:  # first one has no previous one
                                    new_dic[k][new_dic_dr_counter] = disc_rounds[k][l]
                                    new_dic_dr_counter += 1
                                    continue
                                if disc_rounds[k][l][0] - disc_rounds[k][l-1][1] < 60:  # if close by merge
                                    new_dic[k][new_dic_dr_counter - 1][1] = disc_rounds[k][l][1]
                                else:  # if far away not merge
                                    new_dic[k][new_dic_dr_counter] = disc_rounds[k][l]
                                    new_dic_dr_counter += 1


        data = []
        for k in new_dic.keys():
            dic = new_dic[k]
            for l in dic.keys():
                data.append(dic[l])
        # Separate starts and ends
        starts = sorted([point[0] for point in data])
        ends = sorted([point[1] for point in data])

        if starts == [] or ends == []:
            final_list = []

        else:
            # go over starts and ends and identify start and end cluster
            clusters_start = {}
            cluster_id = 1  # Start with cluster 1
            clusters_start[cluster_id] = [starts[0]]

            # Iterate through the numbers to group them into clusters
            for i in range(1, len(starts)):
                # If the current number is within 15 seconds of the last number in the current cluster, add it to the current cluster
                if starts[i] - starts[i-1] <= 15:
                    clusters_start[cluster_id].append(starts[i])
                else:
                    # Otherwise, start a new cluster
                    cluster_id += 1
                    clusters_start[cluster_id] = [starts[i]]


            clusters_end = {}
            cluster_id = 1  # Start with cluster 1
            clusters_end[cluster_id] = [ends[0]]

            # Iterate through the numbers to group them into clusters
            for i in range(1, len(ends)):
                # If the current number is within 15 units of the last number in the current cluster, add it to the current cluster
                if ends[i] - ends[i-1] <= 15:
                    clusters_end[cluster_id].append(ends[i])
                else:
                    # Otherwise, start a new cluster
                    cluster_id += 1
                    clusters_end[cluster_id] = [ends[i]]

            # extract all the clusters that seem likely to be true starts and ends: whenever cluster size is > half the streamers in the lobby
            # for starts extract 25% quantile, for ends 75% quantile
            considered_timestamps = 0  # additionally keep count of timestamps i consider for my final clusters for logging purposes
            likely_starts = []
            critical_number = len(list(disc_rounds.keys())) / 2
            for c in clusters_start.keys():
                if len(clusters_start[c]) > critical_number:
                    likely_starts.append(np.percentile(clusters_start[c], 25))
                    considered_timestamps += len(clusters_start[c])
            likely_ends = []
            for c in clusters_end.keys():
                if len(clusters_end[c]) > critical_number:
                    likely_ends.append(np.percentile(clusters_end[c], 75))
                    considered_timestamps += len(clusters_end[c])


            # Combine starts and ends with labels
            combined = [(start, 'start') for start in likely_starts] + [(end, 'end') for end in likely_ends]

            # Sort combined list by the numeric values
            combined_sorted = sorted(combined, key=lambda x: x[0])

            # Initialize variables
            final_list = []
            last_type = 'end'  # Assume we always start with a 'start' after an 'end'

            for value, type in combined_sorted:
                if type == 'start':
                    if last_type == 'start':  # Previous was also 'start', insert None for missing 'end'
                        final_list[-1].append(None)
                        final_list.append([value])
                    else:
                        final_list.append([value])  # Start a new pair with the 'start'
                else:  # Current is 'end'
                    if last_type == 'end':  # Previous was also 'end', insert None for missing 'start'
                        final_list.append([None, value])
                    else:
                        final_list[-1].append(value)  # Complete the current pair with the 'end'
                last_type = type

            # handle final list being only a start (check if final list is not empty because no detected discussion rounds)
            if len(final_list) != 0:
                if len(final_list[-1]) == 1:
                    final_list[-1].append(None)

            # Go over final list and check for Nones if for their likely value (existing start / end +- 150) is existing in the lobbies clusters
            # it is possible that discussion are shorter when everyone already has voted but rather extract too long than too short discussions
            if list(sessions_discussion_lengths[sessions_discussion_lengths['Session']==s]['Duration of Discussion Rounds'])[0] == '15-120':
                usual_disc_length = 152
            if list(sessions_discussion_lengths[sessions_discussion_lengths['Session']==s]['Duration of Discussion Rounds'])[0] == 90:
                usual_disc_length = 108
            for ind, dr in enumerate(final_list):
                if None in dr:
                    if dr[0] is None:  # check if likely end - usual_disc_length is a cluster
                        proposed_start = dr[1] - usual_disc_length
                        if ind != 0:  # when possible compare if time is realistic considering the already established drs
                            if proposed_start < final_list[ind-1][1]:
                                continue
                            for c in clusters_start.keys():
                                for t in clusters_start[c]:
                                    if abs(t - proposed_start) < 5:  # find cluster which has a timestamp closer than 5 seconds
                                        if len(clusters_start[c]) > 2:
                                            final_list[ind][1] = proposed_start
                                            considered_timestamps += len(clusters_start[c])
                    if dr[1] is None:
                        proposed_end = dr[0] + usual_disc_length
                        if ind != len(final_list) - 1:  # when possible compare if time is realistic considering the already established drs
                            if proposed_end > final_list[ind + 1][0]:
                                continue
                            for c in clusters_end.keys():
                                for t in clusters_end[c]:
                                    if abs(t - proposed_end) < 5:  # find cluster which has a timestamp closer than 5 seconds
                                        if len(clusters_end[c]) > 2:
                                            final_list[ind][1] = proposed_end
                                            considered_timestamps += len(clusters_end[c])

        final_discussion_rounds[s][ses_l] = final_list

# save extracted discussion rounds as file
with open('output_discussion_rounds/final_discussion_rounds_from_pyannote.pkl', 'wb') as file:
    pickle.dump(final_discussion_rounds, file)


# for lobbies with None manual inspection
lobbies_to_extract = []
for s in final_discussion_rounds.keys():
    for l in final_discussion_rounds[s].keys():
        for dr in final_discussion_rounds[s][l]:
            if None in dr:
                lobbies_to_extract.append(f'{s}_L{l}')

lobbies_to_extract = list(set(lobbies_to_extract))

def extract_segment(input_file, output_file, start_time, end_time):
    # start_time and end_time are in seconds
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

for l in lobbies_to_extract:
    # extract from mkv file from first streamer of lobby the lobby mkv
    valid_streamer = False  # some streamers have negative lobby start time
    i = 0
    while not valid_streamer:
        streamer = list(lobbies[l[:13]][int(l[15:])].keys())[i]
        start_time = lobbies[l[:13]][int(l[15:])][streamer][0].total_seconds()
        end_time = lobbies[l[:13]][int(l[15:])][streamer][1].total_seconds()
        if lobbies[l[:13]][int(l[15:])][streamer][0].days < 0:  # take other streamer from lobby
            i += 1
        else:
            valid_streamer = True
    extract_segment(f'../../../../dimstore/pop520978/data/{l[:13]}/{streamer}.mkv', f'{streamer}_L{l[15:]}.mkv', start_time,
                                end_time)


# create excel for easier manual examination
num_columns = 6
column_names = ['Start 1', 'End 1', 'Start 2', 'End 2', 'Start 3', 'End 3']
df = pd.DataFrame(0, index=lobbies_to_extract, columns=column_names)

for s in final_discussion_rounds.keys():
    for l in final_discussion_rounds[s].keys():
        c = 1
        for dr in final_discussion_rounds[s][l]:
            if None in dr:
                if dr[0] is None:
                    df.loc[f'{s}_L{l}', f'Start {c}'] = None
                else:
                    df.loc[f'{s}_L{l}', f'Start {c}'] = str(timedelta(seconds = dr[0]))
                if dr[1] is None:
                    df.loc[f'{s}_L{l}', f'End {c}'] = None
                else:
                    df.loc[f'{s}_L{l}', f'End {c}'] = str(timedelta(seconds = dr[1]))
                c += 1

df.to_excel('ManualExaminationDiscussionRounds.xlsx')

c = 0
for s in final_discussion_rounds.keys():
    for l in final_discussion_rounds[s].keys():
        c += 1

