import pysrt
import os
import re
import wave
import contextlib
import librosa
from pydub import AudioSegment
import soundfile as sf
import math
import numpy as np
import logging

OUTPUT_FILE = 'wrongly_aligned_discussion_rounds.txt'
wrongly_aligned_discussion_rounds = []


os.mkdir('data/audios')
os.mkdir('data/alignments')
partial_transcriptions = os.listdir('../relevant_transcriptions')

alignments = []  # for each transcription the alignment is appended to this list
transcriptions = []  # for each transcription the transcription is appended to this list
for t in partial_transcriptions:

    # for streamer vikramafc sometimes there is silence in the stream, these nine discussion rounds are skipped for him
    if t in ['2022-02-23_S1_l18_1_vikramafc.srt', '2022-02-23_S1_l18_0_vikramafc.srt', '2022-02-02_S1_l9_0_vikramafc.srt', '2022-02-21_S1_l9_0_vikramafc.srt',
             '2022-03-02_S1_l9_1_vikramafc.srt', '2022-03-03_S1_l9_1_vikramafc.srt', '2022-02-04_S1_l11_1_vikramafc.srt', '2022-02-23_S1_l7_0_vikramafc.srt', '2022-02-21_S1_l16_1_vikramafc.srt']:
        continue

    # first create flac file from wav file as this is used in the model architecture later on
    wav_audio = AudioSegment.from_file(f'../relevant_discussion_rounds_audio/{t.replace("srt", "wav")}', format="wav")
    wav_audio.export(f'data/audios/{t.replace("srt", "flac")}', format="flac")

    # from srt file build list of words and list of end timestamps for the corresponding words
    subs = pysrt.open(f'../relevant_transcriptions/{t}')
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
                times.append(str(60 * sub.start.minutes + sub.start.seconds) + '.' + str(sub.start.milliseconds))
        if '</font>' in sub.text or '<font color="#00ff00">' in sub.text:
            word = extract_colored_text(sub.text)[0]
            word = word.replace(' ', '')  # in the rare case multiple words are in font, f.e. on top
            words.append(word.upper())
            times.append(
                str(60 * sub.end.minutes + sub.end.seconds) + '.' + str(sub.end.milliseconds))  # Add end time in seconds to the list
        else:
            times[-1] = str(60 * sub.end.minutes + sub.end.seconds) + '.' + str(sub.end.milliseconds)

    # add silence part at end of utterance to alignments
    data, samplerate = sf.read(f'data/audios/{t.replace("srt", "flac")}')
    length_seconds = len(data) / samplerate
    words.append(',')
    times.append(str(length_seconds))

    # write final alignment with id in alignment list
    # Join words and times with commas
    words_string = ",".join(words)
    times_string = ",".join(times)
    alignment = f'{t[:-4]} "{words_string}" "{times_string}"'
    alignment = alignment.replace(',,', ',')
    alignments.append(alignment)

    # append transcription with id to transcription list
    transcriptions.append(" ".join(words[1:-1]))

# when all transcriptions are aligned and transcribed, create two files storing them
with open(f'data/alignments/alignments.txt', 'w') as file:
    for item in alignments:
        # Write each item on a new line
        file.write(f"{item}\n")

with open(f'data/audios/trans.txt', 'w') as file:
    for item in transcriptions:
        # Write each item on a new line
        file.write(f"{item}\n")


# convert alignments to a combination of 'W' denoting words and ' ' denoting silence.
transcripts = []
with open('data/alignments/alignments.txt') as f:
    for line in f.readlines():
        name = line.split(' ')[0]
        full_path = f'data/audios/{name}.flac'
        aligned_text = line.split(' ')[1][1:-1]
        tstamps = line.split(' ')[2][1:-2]  # without the newline..

        # throw away the actual words if not needed...
        aligned_text = re.sub(r"[A-Z']+", 'W', aligned_text)

        # store the aligned transcript in the list
        transcripts.append((full_path, name, aligned_text, tstamps))

# resample audios to mono stream and 16000 audio sampling rate
def convert_flac_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.flac'):
                print(os.path.join(root, file))
                stereo_audio, samplerate = sf.read(os.path.join(root, file))
                # Convert to mono by averaging the two channels
                mono_audio = np.mean(stereo_audio, axis=1)
                # resample to 16000 audio sampling rate
                audio = librosa.resample(mono_audio, orig_sr=samplerate, target_sr=16000)
                # Export the mono audio to a new FLAC file
                sf.write(os.path.join(root, file), audio, 16000)


convert_flac_files('data/audios')


# --> in transcripts i have alignments of all discussion rounds
with open('data/wav.scp', 'w') as wav_scp, \
            open('data/text', 'w') as text:
    for utt in transcripts:
        data = np.array([])
        full_names = ''
        transcript = ''
        alignment = ''

        tstamps = []
        prev_end_stamp = 0
        only_utterance = False
        x, sr = sf.read(utt[0])
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
                if (end - x.size) > 1:
                    print(f'Size error: {utt[1]}')
                    continue

        tstamps = stamps
        full_names += utt[1]
        transcript += utt[2]

        alignment = ' '.join(tstamps)

        # Check if the lengths of transcript and timestamps match
        transcript_parts = transcript.split(',')
        alignment_parts = alignment.split(' ')
        if len(transcript_parts) != len(alignment_parts):
            wrongly_aligned_discussion_rounds.append(full_names)
            #print(f"Skipping {full_names} due to mismatched lengths of gtruth and timestamps")
            #continue




        # and write an entry to our wav.scp and text files
        wav_scp.write(full_names + ' flac -d -c -s ' + 'data/audios/' +
                      full_names + '.flac |\n')
        text.write(full_names + ' ' + transcript + ' ' + alignment + '\n')



with open(OUTPUT_FILE, 'w') as f:
    # First, log the lengths of all files
    f.write(f'{wrongly_aligned_discussion_rounds}')
