import soundfile as sf
import numpy as np
import os
import librosa


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


root_directory = 'data/LibriSpeech/dev-clean'
convert_flac_files(root_directory)
