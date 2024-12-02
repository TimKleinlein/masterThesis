import os
from pydub import AudioSegment
import pickle

# Load the dictionary from the pickle file
with open('label_dict.pkl', 'rb') as file:
    target_times = pickle.load(file)

# Directory containing the original flac files
input_directory = 'data/audios'
# Directory to save the new flac files
output_directory = 'data/target_audios'
os.makedirs(output_directory, exist_ok=True)

# Iterate over each key in the dictionary
for filename, time_intervals in target_times.items():
    # Load the original audio file
    input_path = os.path.join(input_directory, f"{filename}.flac")
    audio = AudioSegment.from_file(input_path, format="flac")

    # Create a new AudioSegment for the extracted parts
    extracted_audio = AudioSegment.empty()

    # Iterate over the time intervals and extract the corresponding audio segments
    for start, end in time_intervals:
        start_ms = int(start * 1000)  # Convert seconds to milliseconds
        end_ms = int(end * 1000)  # Convert seconds to milliseconds
        extracted_audio += audio[start_ms:end_ms]

    # Save the extracted audio to a new file with the _target suffix
    output_path = os.path.join(output_directory, f"{filename}_target.flac")
    extracted_audio.export(output_path, format="flac")

    print(f"Extracted audio saved to {output_path}")

print("Audio extraction completed.")
