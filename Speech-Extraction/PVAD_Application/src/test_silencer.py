from pydub import AudioSegment
import time

start = time.time()

# Load the audio file
audio = AudioSegment.from_file("data/audios/2022-02-19_S1_l11_0_ozzaworld.flac", format="flac")

# Define the timestamps to mute (in milliseconds)
timestamps = [(2000, 4000), (8200, 10000), (11000, 13000), (15000, 16000), (20000, 40000), (41000, 50000)]

# Mute the specified portions
for start, end in timestamps:
    audio = audio[:start] + AudioSegment.silent(duration=(end - start)) + audio[end:]

# Export the edited audio
audio.export("silenced.flac", format="flac")

end = time.time()
print(f'Executed in {end - start}')
