import stable_whisper
import pysrt
import os
import re
import wave
import contextlib
from pydub import AudioSegment
import soundfile as sf



model = stable_whisper.load_model('large')

streamers = os.listdir('train_data_diarization')
for s in streamers:
    os.mkdir(f'train_data_diarization/{s}/utterances_srt_files')
    utterances = os.listdir(f'train_data_diarization/{s}/utterances')
    alignments = []  # for each utterance the alignment is appended to this list
    transcriptions = []  # for each utterance the transcription is appended to this list
    # for each utterances transcription that is written in srt file
    for u in utterances:
        # first create flac file from wav file as this is used in the model architecture later on
        wav_audio = AudioSegment.from_file(f'train_data_diarization/{s}/utterances/{u}', format="wav")
        wav_audio.export(f'train_data_diarization/{s}/utterances/{u[:-4]}.flac', format="flac")

        # then transcription
        result = model.transcribe(f'train_data_diarization/{s}/utterances/{u}')
        result.to_srt_vtt(f'train_data_diarization/{s}/utterances_srt_files/{u[:-4]}.srt')

        # from the srt file build list of words and list of end timestamps for the corresponding words
        subs = pysrt.open(f'train_data_diarization/{s}/utterances_srt_files/{u[:-4]}.srt')

        # if subs is empty because short utterance delete utterance and continue
        if len(subs) == 0:
            os.remove(f'train_data_diarization/{s}/utterances/{u}')
            os.remove(f'train_data_diarization/{s}/utterances/{u[:-4]}.flac')
            os.remove(f'train_data_diarization/{s}/utterances_srt_files/{u[:-4]}.srt')
            continue

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
                words.append(word.upper())
                times.append(
                    str(sub.end.seconds) + '.' + str(sub.end.milliseconds))  # Add end time in seconds to the list
            else:
                times[-1] = str(sub.end.seconds) + '.' + str(sub.end.milliseconds)

        # add silence part at end of utterance to alignments
        data, samplerate = sf.read(f'train_data_diarization/{s}/utterances/{u[:-4]}.flac')
        length_seconds = len(data) / samplerate
        words.append(',')
        times.append(str(length_seconds))

        # write final alignment with id in alignment list
        # Join words and times with commas
        words_string = ",".join(words)
        times_string = ",".join(times)
        alignment = f'{s}-{u[:-4]} "{words_string}" "{times_string}"'
        alignment = alignment.replace(',,', ',')
        alignments.append(alignment)

        # append transcription with id to transcription list
        transcriptions.append(" ".join(words[1:-1]))

    # when all utterances of streamer are aligned and transcribed, create two files storing them
    with open(f'train_data_diarization/{s}/{s}.alignments.txt', 'w') as file:
        for item in alignments:
            # Write each item on a new line
            file.write(f"{item}\n")

    with open(f'train_data_diarization/{s}/{s}.trans.txt', 'w') as file:
        for item in transcriptions:
            # Write each item on a new line
            file.write(f"{item}\n")
