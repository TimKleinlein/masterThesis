# manually inspected all lobbies of a streamer where role assignment was None, Spy, or Seer.
# Labeled according to following metric suggested by the images:
# - Delete: Lobby start was the wrong timestamp --> exclude this lobby for all streamers from analysis
# - Delete -I: Lobby start is wrong timestamp for this streamer only (maybe streamer joined lobby late etc.) --> exclude this lobby for this streamer from analysis
# - Later: Lobby start is too early --> extract additional images a few seconds later for all streamers of lobby
# - Later -I: Lobby start is too early --> extract additional images a few seconds later for this streamers only
# - Role name: for this lobby and this streamer the correct role was extracted and should now be assigned to him
import os
import pandas as pd
import pickle as pkl
import ffmpeg

manual_examination_results = pd.read_excel('ManualExamination.xlsx')

with open('identified_roles.pkl', 'rb') as file:
    identified_roles = pkl.load(file)


with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobbies = pkl.load(file)


# reset role for all streamer lobbies where i have manual examination result
for sl in manual_examination_results['StreamerLobby']:
    ses, l, streamer = sl.split(' - ')[0], sl.split(' - ')[1], sl.split(' - ')[2]
    if manual_examination_results[manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower() not in ['delete', 'delete -i', 'later', 'later -i']:

        identified_roles[ses][l][streamer] = manual_examination_results[manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower()

# delete all lobbies for individual streamer or for all streamers if i detected in manual role examination that lobby start timestamp is off
for sl in manual_examination_results['StreamerLobby']:
    ses, l, streamer = sl.split(' - ')[0], sl.split(' - ')[1], sl.split(' - ')[2]
    if manual_examination_results[manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower() == 'delete':
        if l in identified_roles[ses]:
            del(identified_roles[ses][l])
    if manual_examination_results[manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower() == 'delete -i':
        if streamer in identified_roles[ses][l]:
            del(identified_roles[ses][l][streamer])



# Save the updated dictionary of identified roles
with open('identified_roles.pkl', 'wb') as f:
    pkl.dump(identified_roles, f)


# extract extra images for role detection for either individual streamer or all streamers in lobbies where current images are too early

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

os.mkdir('additional_images')

for sl in manual_examination_results['StreamerLobby']:
    ses, l, streamer = sl.split(' - ')[0], sl.split(' - ')[1], sl.split(' - ')[2]
    if manual_examination_results[manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower() == 'later - i':
        os.makedirs(f'additional_images/{ses}/{l}/{streamer}')
        start_timestamp = lobbies[ses][int(l)][streamer][0].total_seconds()
        video_path = f'../../pop520978/data/{ses}/{streamer}.mkv'
        t_options = list(range(int(start_timestamp) + 12, int(start_timestamp) + 60))
        for t_option in t_options:
            output_path = f'additional_images/{ses}/{l}/{streamer}/{t_option}.jpg'
            extract_frame(video_path, t_option, output_path)
    elif manual_examination_results[manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower() == 'later':
        all_streamers_lobby = os.listdir(f'images/{ses}/{l}')
        for j in all_streamers_lobby:
            if os.path.isdir(f'additional_images/{ses}/{l}/{j}'):
                continue
            else:
                os.makedirs(f'additional_images/{ses}/{l}/{j}')
                start_timestamp = lobbies[ses][int(l)][j][0].total_seconds()
                video_path = f'../../pop520978/data/{ses}/{j}.mkv'
                t_options = list(range(int(start_timestamp) + 12, int(start_timestamp) + 60))
                for t_option in t_options:
                    output_path = f'additional_images/{ses}/{l}/{j}/{t_option}.jpg'
                    extract_frame(video_path, t_option, output_path)

