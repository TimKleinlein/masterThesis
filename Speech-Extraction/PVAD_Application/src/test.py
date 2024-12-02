import pysrt
import os
import re
import wave
import contextlib
from pydub import AudioSegment
import soundfile as sf
import math
import numpy as np
import pickle as pkl

# two parameters:
# ratio when label 2 is counted
# ratio how much of sequence has to be labeled with such a 2 to be extracted
# i try three extraction modes: low_acc(0.5, 0.6), mid_acc(0.8, 0.6), high_acc(0.8, 0.8)

subs = pysrt.open('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/2022-02-04_S1_l9_0_ozzaworld.srt')
for sub in subs:
    print(sub.start)
    print(sub.text)
    print(sub.end)
    print()


with open(f'/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/pvad_output_files/2022-02-04_S1_l9_0_ozzaworld_labeled_intervals.pkl', 'rb') as file:
    intervals = pkl.load(file)

final_dic = {}

for threshold in [0.0, 0.5, 0.7, 0.8]:
    sequence = []
    timestamps = []

    prev_text = ''

    for sub in subs:
        if sub.text_without_tags == prev_text:
            timestamps[-1][1] = sub.end
        else:
            sequence.append(sub.text_without_tags)
            timestamps.append([sub.start, sub.end])
            prev_text = sub.text_without_tags


    target_sequences = []
    for ind, s in enumerate(sequence):
        sequence_start = timestamps[ind][0].minutes * 60 + timestamps[ind][0].seconds * 1 + timestamps[ind][0].milliseconds * 0.001
        sequence_end = timestamps[ind][1].minutes * 60 + timestamps[ind][1].seconds * 1 + timestamps[ind][1].milliseconds * 0.001

        # check if over 80% are labeled with 2
        sequence_labels = []

        for i in intervals.keys():
            if i[0] >= sequence_start and i[1] <= sequence_end:
                if intervals[i].index(max(intervals[i])) == 2 and max(intervals[i]) > threshold:
                    sequence_labels.append(2)
                else:
                    sequence_labels.append(0)

        ratio_target = sequence_labels.count(2) / len(sequence_labels)

        if ratio_target > 0.6:
            target_sequences.append(s)

    combined_target_sequences = []
    prev_end = - 1000
    for ts in target_sequences:
        if ((timestamps[sequence.index(ts)][0] - prev_end).seconds + (timestamps[sequence.index(ts)][0] - prev_end).milliseconds / 1000) < 2:
            combined_target_sequences[-1] = combined_target_sequences[-1] + ' ' + ts
            prev_end = timestamps[sequence.index(ts)][1]
        else:
            combined_target_sequences.append(ts)
            prev_end = timestamps[sequence.index(ts)][1]


    final_dic[threshold] = combined_target_sequences



















for f in os.listdir('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/partial_relevant_transcriptions'):
    subs = pysrt.open(f'/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/partial_relevant_transcriptions/{f}')
    print(f)
    for index, sub in enumerate(subs):
        if index == len(subs) - 1:
            print(sub.end)
    print('----')



# Initialize variables to store the words and their corresponding end times
words = [","]  # to account for silence before first word
times = [str(subs[0].start.seconds) + '.' + str(subs[0].start.milliseconds)]

# Process each word
def extract_colored_text(input_string):
    # Regular expression pattern to find text between <font color="#00ff00"> and </font>
    pattern = r'<font color="#00ff00">(.*?)</font>'
    # Find all matches
    matches = re.findall(pattern, input_string)
    cleaned_words = [re.sub(r'[^\w\s]', '', match) for match in matches]

    return cleaned_words

for index, sub in enumerate(subs):
    if index != 0:
        if sub.start > subs[index - 1].end:  # in that case add missing silence
            words.append(",")
            times.append(str(sub.start.seconds) + '.' + str(sub.start.milliseconds))
    if '</font>' in sub.text or '<font color="#00ff00">' in sub.text:
        word = extract_colored_text(sub.text)[0]
        word = word.replace(' ', '')  # in the rare case multiple words are in font, f.e. on top
        words.append(word.upper())
        times.append(
            str(sub.end.seconds) + '.' + str(sub.end.milliseconds))  # Add end time in seconds to the list
    else:
        times[-1] = str(sub.end.seconds) + '.' + str(sub.end.milliseconds)

# add silence part at end of utterance to alignments
data, samplerate = sf.read('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/2022-02-22_S1_l16_4_skadj.flac')
length_seconds = len(data) / samplerate
words.append(',')
times.append(str(length_seconds))
alignments = []

# write final alignment with id in alignment list
# Join words and times with commas
words_string = ",".join(words)
times_string = ",".join(times)
alignment = f'name "{words_string}" "{times_string}"'
alignment = alignment.replace(',,', ',')
alignments.append(alignment)


# when all transcriptions are aligned and transcribed, create two files storing them
with open(f'/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/alignments.txt', 'w') as file:
    for item in alignments:
        # Write each item on a new line
        file.write(f"{item}\n")


with open('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/alignments.txt') as f:
    for line in f.readlines():
        print(len((line.split(' ')[2][1:-2]).split(',')))
        print(len((line.split(' ')[1][1:-1]).split(',')))
        name = line.split(' ')[0]
        full_path = f'data/audios/{name}.flac'
        aligned_text = line.split(' ')[1][1:-1]
        tstamps = line.split(' ')[2][1:-2]  # without the newline..

        # throw away the actual words if not needed...
        aligned_text = re.sub(r"[A-Z']+", 'W', aligned_text)

transcripts = []
transcripts.append((full_path, name, aligned_text, tstamps))

for utt in transcripts:
    full_names = ''
    transcript = ''
    alignment = ''

    tstamps = []
    prev_end_stamp = 0
    only_utterance = False
    stamps = utt[3].split(',')

    tstamps = stamps
    full_names += utt[1]
    transcript += utt[2]

    alignment = ' '.join(tstamps)

    # Check if the lengths of transcript and timestamps match
    transcript_parts = transcript.split(',')
    alignment_parts = alignment.split(' ')
    if len(transcript_parts) != len(alignment_parts):
        print(f"Skipping {full_names} due to mismatched lengths of gtruth and timestamps")
        continue


for utt in transcripts:
    data = np.array([])
    full_names = ''
    transcript = ''
    alignment = ''

    tstamps = []
    prev_end_stamp = 0
    only_utterance = False
    x, sr = sf.read('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/2022-02-22_S1_l16_4_skadj.flac')
    assert (sr == 16000), f'Invalid source audio sample rate {sr}'
    stamps = utt[3].split(',')
    try:
        check_float = float(stamps[-1])
    except ValueError:  # alignment file is wrong for this line
        print(f'Float conversion failed for {utt[1]}')
        continue

    # check if size error
    end_stamp = math.trunc(float(stamps[-1]) * 100) / 100
    end = end_stamp * sr
    if x.size != end:
        if x.size <= end:
            print(f'Size error: {utt[1]}')
            continue



from pydub import AudioSegment

# Load the audio file
audio = AudioSegment.from_file("path/to/your/file.flac", format="flac")

# Define the timestamps to mute (in milliseconds)
timestamps = [(2000, 4000), (8200, 10000)]

# Mute the specified portions
for start, end in timestamps:
    audio = audio[:start] + AudioSegment.silent(duration=(end - start)) + audio[end:]

# Export the edited audio
audio.export("path/to/your/output_file.flac", format="flac")
