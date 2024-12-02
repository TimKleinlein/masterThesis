import ffmpeg
import os
import pickle

with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobbies = pickle.load(file)

relevant_streamers = os.listdir('../Speech-Extraction/Personal_VAD_Model/data/LibriSpeech/dev-clean')
if '.DS_Store' in relevant_streamers:
    relevant_streamers.remove('.DS_Store')

sessions = [
 '2022-01-26_S1', '2022-01-27_S1', '2022-01-28_S1', '2022-02-01_S1', '2022-02-02_S1', '2022-02-04_S1', '2022-02-05_S1',
 '2022-02-08_S1', '2022-02-09_S1', '2022-02-10_S1', '2022-02-12_S1', '2022-02-15_S1', '2022-02-16_S1', '2022-02-17_S1', '2022-02-19_S1',
 '2022-02-21_S1', '2022-02-22_S1', '2022-02-23_S1', '2022-02-24_S1', '2022-02-26_S1', '2022-03-01_S1', '2022-03-02_S1', '2022-03-03_S1',
 '2022-03-09_S1', '2022-03-10_S1', '2022-05-24_S1', '2022-05-24_S2']


def extract_middle_part(string):
    parts = string.split('_')
    if len(parts) > 2:
        return parts[2]
    else:
        return None

def extract_frame(video_path, timestamp, output_path):
    """
    Extract a frame from the video at a specific timestamp.

    :param video_path: Path to the input video file.
    :param timestamp: Timestamp to extract the frame at (in seconds).
    :param output_path: Path to save the extracted frame.
    """
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    try:
        (
            ffmpeg
                .input(video_path, ss=timestamp)
                .filter('scale', '1280', '720')
                .output(output_path, vframes=1, qscale=6)
                .run()
        )
        print(f"Frame extracted and saved to {output_path}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")


os.mkdir('images')

for ses in sessions:
    os.mkdir(f'images/{ses}')
    for lob in lobbies[ses].keys():
        os.mkdir(f'images/{ses}/{lob}')
        for streamer in lobbies[ses][lob].keys():
            if extract_middle_part(streamer) not in relevant_streamers:
                continue
            else:
                os.mkdir(f'images/{ses}/{lob}/{streamer}')
                start_timestamp = lobbies[ses][lob][streamer][0].total_seconds()
                video_path = f'../../pop520978/data/{ses}/{streamer}.mkv'
                t_options = list(range(int(start_timestamp) - 5, int(start_timestamp) + 11))

                for t_option in t_options:
                    output_path = f'images/{ses}/{lob}/{streamer}/{t_option}.jpg'
                    extract_frame(video_path, t_option, output_path)
