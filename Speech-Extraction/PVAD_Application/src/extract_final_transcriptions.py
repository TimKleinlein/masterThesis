import pysrt
import os
import re
import soundfile as sf
import math
import numpy as np
import pickle as pkl

# extracting relevant srt files in following way: Go over total srt file entries and build sequences (sentences) by
# combining all sequences as ts-whisper would have done it (for this sequence define start and end time according to first and last word)
# then check for each sequence whether it belongs to target speaker or not by going over all the labels of all intervals
# of the sequence. Then check if more than a specific ratio of all labels of sequence are 2. Two parameters to define:
# probability of label 2 assignment such that label 2 assignment is counted
# ratio how much of sequence has to be labeled with such a 2 to be extracted
# i try three extraction modes: low_acc(0.5, 0.6), mid_acc(0.8, 0.6), high_acc(0.8, 0.8)
# finally combine all sequences to one concatenated sequence if their end / start are within 2 seconds

discussion_rounds = os.listdir('data/pvad_output_files')
discussion_rounds = [x for x in discussion_rounds if 'nontarget_label_dict' not in x]
discussion_rounds = [x.replace('_labeled_intervals.pkl', '') for x in discussion_rounds]


final_transcriptions = {}
final_transcriptions_with_timestamps = {}

for dr in discussion_rounds:

    final_transcriptions[dr] = {}
    final_transcriptions_with_timestamps[dr] = {}
    subs = pysrt.open(
        f'../relevant_transcriptions/{dr}.srt')
    with open(
            f'data/pvad_output_files/{dr}_labeled_intervals.pkl',
            'rb') as file:
        intervals = pkl.load(file)

    for threshold in [('low_acc', 0.5, 0.6)]:#, ('mid_acc', 0.8, 0.6), ('high_acc', 0.8, 0.8)]:
        sequences = []
        timestamps = []

        prev_text = ''

        for sub in subs:
            if sub.text_without_tags == prev_text:
                timestamps[-1][1] = sub.end
            else:
                sequences.append(sub.text_without_tags)
                timestamps.append([sub.start, sub.end])
                prev_text = sub.text_without_tags

        target_sequences = []
        for ind, s in enumerate(sequences):
            sequence_start = timestamps[ind][0].minutes * 60 + timestamps[ind][0].seconds * 1 + timestamps[ind][
                0].milliseconds * 0.001
            sequence_end = timestamps[ind][1].minutes * 60 + timestamps[ind][1].seconds * 1 + timestamps[ind][
                1].milliseconds * 0.001

            # check if over 80% are labeled with 2
            sequence_labels = []

            for i in intervals.keys():
                if i[0] >= sequence_start and i[1] <= sequence_end:
                    if intervals[i].index(max(intervals[i])) == 2 and max(intervals[i]) > threshold[1]:
                        sequence_labels.append(2)
                    else:
                        sequence_labels.append(0)

            ratio_target = sequence_labels.count(2) / len(sequence_labels)

            if ratio_target > threshold[2]:
                target_sequences.append(s)

        combined_target_sequences = []
        combined_target_sequences_with_timestamps = []
        for ts_ind, ts in enumerate(target_sequences):
            if ts_ind == 0:
                combined_target_sequences.append(ts)
                prev_end = timestamps[sequences.index(ts)][1]
                start = timestamps[sequences.index(ts)][0]
                end = timestamps[sequences.index(ts)][1]
                combined_target_sequences_with_timestamps.append((ts, start, end))
                continue
            if ((timestamps[sequences.index(ts)][0] - prev_end).seconds + (
                    timestamps[sequences.index(ts)][0] - prev_end).milliseconds / 1000) < 2:
                combined_target_sequences[-1] = combined_target_sequences[-1] + ' ' + ts
                prev_end = timestamps[sequences.index(ts)][1]
                end = timestamps[sequences.index(ts)][1]
                combined_target_sequences_with_timestamps[-1] = (combined_target_sequences_with_timestamps[-1][0] + ' ' + ts, combined_target_sequences_with_timestamps[-1][1], end)
            else:
                combined_target_sequences.append(ts)
                prev_end = timestamps[sequences.index(ts)][1]
                start = timestamps[sequences.index(ts)][0]
                end = timestamps[sequences.index(ts)][1]
                combined_target_sequences_with_timestamps.append((ts, start, end))

        final_transcriptions[dr][threshold[0]] = combined_target_sequences
        final_transcriptions_with_timestamps[dr][threshold[0]] = combined_target_sequences_with_timestamps


# save final dictionary
#with open('data/final_transcriptions.pkl', 'wb') as f:
#    pkl.dump(final_transcriptions, f)

with open('data/final_transcriptions_with_timestamps.pkl', 'wb') as f:
    pkl.dump(final_transcriptions_with_timestamps, f)
