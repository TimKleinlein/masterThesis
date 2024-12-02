from moviepy.editor import VideoFileClip
import subprocess

def extract_segment_with_ffmpeg(input_file, output_file, start_time, end_time):
    try:
        # Convert start_time and end_time to the format HH:MM:SS.MS
        start_time_str = "{:02}:{:02}:{:06.3f}".format(
            int(start_time // 3600),
            int((start_time % 3600) // 60),
            start_time % 60
        )
        end_time_str = "{:02}:{:02}:{:06.3f}".format(
            int(end_time // 3600),
            int((end_time % 3600) // 60),
            end_time % 60
        )

        # Use ffmpeg with re-encoding
        command = [
            'ffmpeg', '-y', '-i', input_file,
            '-ss', start_time_str, '-to', end_time_str,
            '-c:v', 'libx264', '-c:a', 'aac',  # Re-encoding the video and audio
            output_file
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = '../../pop520978/data/2022-02-23_S1/2022-02-23_S1_vikramafc_1307860812.mkv'
output_file = 'extract.mkv'
start_time = 38.456574
end_time = 40

extract_segment_with_ffmpeg(input_file, output_file, start_time, end_time)

def get_mkv_duration_in_milliseconds(file_path):
    try:
        clip = VideoFileClip(file_path)
        duration_in_seconds = clip.duration
        duration_in_ms = int(duration_in_seconds * 1000)
        return duration_in_ms
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Get the duration of the extracted segment
duration_in_ms = get_mkv_duration_in_milliseconds(output_file)
if duration_in_ms is not None:
    print(f"The duration of the file is {duration_in_ms} milliseconds.")
