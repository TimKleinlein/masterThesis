import os
import shutil
import random


def collect_all_files(base_dir):
    all_files = []
    for streamer in os.listdir(base_dir):
        streamer_path = os.path.join(base_dir, streamer)
        if os.path.isdir(streamer_path):
            subdir = "1"
            subdir_path = os.path.join(streamer_path, subdir)
            if os.path.isdir(subdir_path):
                files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.flac')]
                all_files.extend(files)
    return all_files


def copy_files_to_target(files, base_dir, target_dir):
    for file_path in files:
        relative_path = os.path.relpath(file_path, base_dir)
        target_path = os.path.join(target_dir, relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(file_path, target_path)


def split_dataset(base_dir, train_dir, test_dir, train_ratio=0.8):
    # Collect all .flac files from the base directory
    all_files = collect_all_files(base_dir)

    # Shuffle and split the files
    random.shuffle(all_files)
    split_point = int(len(all_files) * train_ratio)
    train_files = all_files[:split_point]
    test_files = all_files[split_point:]

    # Copy files to the respective directories
    copy_files_to_target(train_files, base_dir, train_dir)
    copy_files_to_target(test_files, base_dir, test_dir)


# Set your base directory
base_directory = "/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model/data/LibriSpeech/dev-clean"
train_directory = "/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model/data/LibriSpeech/dev-clean-train"
test_directory = "/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model/data/LibriSpeech/dev-clean-test"

# Call the function
split_dataset(base_directory, train_directory, test_directory)
