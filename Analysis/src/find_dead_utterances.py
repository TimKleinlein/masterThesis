import json
import datasets
import os
from transformers import AutoTokenizer
import pickle as pkl
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Set the base output directory
base_output_dir = "data/bert_classifier"

# Load the tokenized dataset for low accuracy mode
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Load the transcriptions data
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# Recreate the data processing as in your training script
data = []
groups = []
for dr in transcriptions.keys():
    for ind, utt in enumerate(transcriptions[dr]["low_acc"]):
        data.append({'text': utt, 'label': transcriptions[dr]['role'], 'group': dr})

# Apply the role mapping as in your training script
ROLES_ToR = {
    "CREWMATE": ["crewmate", "mayor", "medium", "swapper", "time master", "engineer", "sheriff", "deputy",
                 "lighter", "detective", "medic", "seer", "hacker", "tracker", "snitch", "spy", "portalmaker",
                 "security guard", "medium", "trapper", "nice guesser", "bait", "shifter"],
    "NEUTRAL": ["neutral", "jester", "arsonist", "jackal", "sidekick", "vulture", "lawyer", "pursuer", "thief"],
    "IMPOSTOR": ["impostor", "godfather", "mafioso", "janitor", "morphling", "camouflager", "vampire", "eraser",
                 "trickster", "cleaner", "warlock", "bounty hunter", "witch", "ninja", "bomber", "yo-yo",
                 "evil guesser"]
}

ROLES_ToU = {
    "CREWMATE": ["crewmate", "detective", "haunter", "investigator", "mystic", "seer", "snitch", "spy", "tracker",
                 "trapper", "sheriff", "veteran", "vigilante", "altruist", "medic", "engineer", "mayor", "medium",
                 "swapper", "transporter", "aurial", "hunter", "imitator", "oracle", "vampire hunter", "time lord"],
    "NEUTRAL": ["neutral", "amnesiac", "guardian angel", "survivor", "executioner", "jester", "phantom", "arsonist",
                "plaguebearer", "the glitch", "werewolf", "doomsayer", "juggernaut"],
    "IMPOSTOR": ["impostor", "grenadier", "morphling", "swooper", "traitor", "blackmailer", "janitor", "miner",
                 "undertaker", "bomber", "escapist", "venerer", "warlock", "poisoner", "underdog"]
}

for entry in data:
    found = False
    for r in ROLES_ToR.keys():
        for sr in ROLES_ToR[r]:
            if sr == entry['label']:
                entry['label'] = r
                found = True
                break
    if not found:
        for r in ROLES_ToU.keys():
            for sr in ROLES_ToU[r]:
                if sr == entry['label']:
                    entry['label'] = r
                    found = True
                    break

# Remove neutral roles from data
data = [entry for entry in data if entry['label'] != 'NEUTRAL']

texts = [entry['text'] for entry in data]
labels = [0 if entry['label'] == 'CREWMATE' else 1 for entry in data]
groups = [entry['group'] for entry in data]

# Recreate the dataset
dataset = datasets.Dataset.from_dict({'text': texts, 'label': labels, 'group': groups})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the test indices for low accuracy and fold 1
test_indices_path = os.path.join(base_output_dir, "low_acc", "test_indices_fold_1.json")
with open(test_indices_path, 'r') as f:
    test_indices = json.load(f)

# Extract the discussion rounds from the test dataset
test_discussion_rounds = [groups[idx] for idx in test_indices]

# Get the unique discussion rounds
unique_test_discussion_rounds = set(test_discussion_rounds)

# Save the unique discussion rounds to a file
output_discussion_rounds_path = os.path.join("data", "test_discussion_rounds_low_acc_fold_1.txt")
with open(output_discussion_rounds_path, 'w') as f:
    for dr in unique_test_discussion_rounds:
        f.write(f"{dr}\n")

print(f"Unique discussion rounds saved to {output_discussion_rounds_path}")

for dr in unique_test_discussion_rounds:
    print(dr)

# now select 30 random discussion rounds with highest probability to be from dead people and extract them
discussion_rounds = []

# Open the file and read each line
with open('data/test_discussion_rounds_low_acc_fold_1.txt', 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace (including newline characters) and add to the list
        discussion_rounds.append(line.strip())

dr_number = [x.split('_')[3] for x in discussion_rounds]
top_30_indices = np.argsort(dr_number)[-30:][::-1]

extract_dr_list = [discussion_rounds[i] for i in top_30_indices]

with open('../Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl', 'rb') as file:
    lobbies = pkl.load(file)

def extract_segment(input_file, output_file, start_time, end_time):
    # start_time and end_time are in seconds
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)


for disc_round in extract_dr_list:
    ses = disc_round.split('_')[0] + '_' + disc_round.split('_')[1]
    lob = int(disc_round.split('_')[2][1:])
    streamer_short = disc_round.split('_')[4]
    for s in lobbies[ses][lob].keys():
        if streamer_short in s:
            streamer = s

    # extract relevant lobby
    lobby_start = lobbies[ses][lob][streamer][0].total_seconds()
    lobby_end = lobbies[ses][lob][streamer][1].total_seconds()

    extract_segment(f'../../pop520978/data/{ses}/{streamer}.mkv',
                    f'data/evaluate_dead_videos/{disc_round}.mkv', lobby_start,
                    lobby_end)
