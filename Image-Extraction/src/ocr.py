import os
import pandas as pd
import cv2
import pytesseract
from pathlib import Path
import numpy as np
import statistics
import pickle
import torch
from collections import Counter

# Set the visible CUDA devices to the second GPU (index 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

session_to_game = pd.read_excel('../Lobby-Synchronization/data/streams_metadata/PlayedGames.xlsx')
session_to_game['Session'] = session_to_game['Session'].apply(lambda x: str(x).replace('\xa0', ''))
df_data = session_to_game.drop(columns=['Session'])
session_to_game['ActiveRole'] = df_data.apply(pd.Series.idxmax, axis=1)

sessions = [
 '2022-01-26_S1', '2022-01-27_S1', '2022-01-28_S1', '2022-02-01_S1', '2022-02-02_S1', '2022-02-04_S1', '2022-02-05_S1',
 '2022-02-08_S1', '2022-02-09_S1', '2022-02-10_S1', '2022-02-12_S1', '2022-02-15_S1', '2022-02-16_S1', '2022-02-17_S1', '2022-02-19_S1',
 '2022-02-21_S1', '2022-02-22_S1', '2022-02-23_S1', '2022-02-24_S1', '2022-02-26_S1', '2022-03-01_S1', '2022-03-02_S1', '2022-03-03_S1',
 '2022-03-09_S1', '2022-03-10_S1', '2022-05-24_S1', '2022-05-24_S2']

# OCR info for detecting the role
OCR_INFO_ROLE = {
    "coords": [240, 25, 1100, 300],  # Adjust these coordinates based on your frame
    "threshold": 10,
    "min_length": 5
}

# Configuration for pytesseract
TESSERACT_CONFIG = r'--oem 3 --psm 6'


def get_thresh_img(img, threshold=0, max_value=255, type=cv2.THRESH_OTSU):
    thresh = cv2.threshold(img, threshold, max_value, type)[1]
    return thresh


def get_open_img(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open_img


def get_image_data(img):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def detect_role(img):
    x1, y1, x2, y2 = OCR_INFO_ROLE["coords"]
    cropped_img = img[y1:y2, x1:x2]
    threshold = get_thresh_img(cropped_img, threshold=OCR_INFO_ROLE["threshold"], type=cv2.THRESH_BINARY)
    open_img = get_open_img(threshold, kernel_size=8)
    inverse = get_thresh_img(open_img, threshold=50, type=cv2.THRESH_BINARY_INV)
    data = get_image_data(inverse)
    your_role = "your role is"
    found_text = ""
    role_detected = None

    for i in range(len(data["text"])):
        if float(data["conf"][i]) > 0.5:  # Confidence threshold
            found_text += data["text"][i] + " "

    if found_text:
        found_text = found_text.replace('Your role is ', '')
        candidates = []
        for role_category, roles in ROLES.items():
            for role in roles:
                distance = levenshtein_distance(role, found_text.lower())
                if distance < 4:
                    candidates.append((role, distance))

        if candidates:

            #  find most likely role
            top_c = candidates[0][0]
            top_d = candidates[0][1]
            for c in candidates:
                if top_d > c[1]:
                    top_c = c[0]
                    top_d = c[1]

            return top_c


identified_roles = {}

for ses in sessions:

    identified_roles[ses] = {}

    # Define roles depending on played game
    if session_to_game[session_to_game['Session'] == ses]['ActiveRole'].values[0].split(' ')[0] == 'TheOtherRoles':
        ROLES = {
            "CREWMATE": ["crewmate", "mayor", "medium", "swapper", "time master", "engineer", "sheriff", "deputy", "lighter", "detective", "medic", "seer", "hacker", "tracker", "snitch", "spy", "portalmaker", "security guard", "medium", "trapper", "nice guesser", "bait", "shifter"],
            "NEUTRAL": ["neutral", "jester", "arsonist", "jackal", "sidekick", "vulture", "lawyer", "prosecutor", "pursuer", "thief"],
            "IMPOSTOR": ["impostor", "godfather", "mafioso", "janitor", "morphling", "camouflager", "vampire", "eraser", "trickster", "cleaner", "warlock", "bounty hunter", "witch", "ninja", "bomber", "yo-yo", "evil guesser"]
        }

    elif session_to_game[session_to_game['Session'] == ses]['ActiveRole'].values[0].split(' ')[0] == 'TownOfUs':
        ROLES = {
            "CREWMATE": ["crewmate", "detective", "haunter", "investigator", "mystic", "seer", "snitch", "spy", "tracker", "trapper", "sheriff", "veteran", "vigilante", "altruist", "medic", "engineer", "mayor", "medium", "swapper", "transporter", "aurial", "hunter", "imitator", "oracle", "prosecutor", "vampire hunter", "time lord"],
            "NEUTRAL": ["neutral", "amnesiac", "guardian angel", "survivor", "executioner", "jester", "phantom", "arsonist", "plaguebearer", "the glitch", "werewolf", "doomsayer", "juggernaut", "vampire"],
            "IMPOSTOR": ["impostor", "grenadier", "morphling", "swooper", "traitor", "blackmailer", "janitor", "miner", "undertaker", "bomber", "escapist", "venerer", "warlock", "poisoner", "underdog"]
        }

    lobbies = os.listdir(f'images/{ses}')
    for lob in lobbies:
        identified_roles[ses][lob] = {}
        streamers = os.listdir(f'images/{ses}/{lob}')
        for streamer in streamers:
            images = os.listdir(f'images/{ses}/{lob}/{streamer}')
            images.sort(key=lambda x: int(x.split('.')[0]))
            role_counter = []
            for i in images:
                frame_path = f'images/{ses}/{lob}/{streamer}/{i}'  # Path to your frame image
                if not Path(frame_path).exists() or i == '.DS_Store':
                    print(f"Frame file {frame_path} does not exist.")
                else:
                    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    role = detect_role(img)
                    if role:
                        role_counter.append(role)
            if role_counter:
                frequency = Counter(role_counter)
                if frequency.most_common()[0][0] != 'spy':
                    identified_roles[ses][lob][streamer] = frequency.most_common()[0][0]
                else:
                    # check if other roles detected, if not assume spy is indeed correct role
                    if len(frequency.most_common()) == 1:
                        identified_roles[ses][lob][streamer] = frequency.most_common()[0][0]
                    else:
                        identified_roles[ses][lob][streamer] = frequency.most_common()[1][0]
            else:
                identified_roles[ses][lob][streamer] = None


# Save the label dictionary to a file
with open('identified_roles.pkl', 'wb') as f:
    pickle.dump(identified_roles, f)
