import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load the transcription data
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# Define the paths for saving plots
save_dir = '/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/'

# Set the style
plt.style.use('seaborn-deep')

# General settings for plots
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.fontsize': 10,
    'axes.grid': False,
    'grid.alpha': 0.5,
})

# Define roles depending on played game
ROLES_ToR = {
    "CREWMATE": ["crewmate", "mayor", "medium", "swapper", "time master", "engineer", "sheriff", "deputy", "lighter", "detective", "medic", "seer", "hacker", "tracker", "snitch", "spy", "portalmaker", "security guard", "medium", "trapper", "nice guesser", "bait", "shifter"],
    "NEUTRAL": ["neutral", "jester", "arsonist", "jackal", "sidekick", "vulture", "lawyer", "pursuer", "thief"],
    "IMPOSTOR": ["impostor", "godfather", "mafioso", "janitor", "morphling", "camouflager", "vampire", "eraser", "trickster", "cleaner", "warlock", "bounty hunter", "witch", "ninja", "bomber", "yo-yo", "evil guesser"]
}

ROLES_ToU = {
    "CREWMATE": ["crewmate", "detective", "haunter", "investigator", "mystic", "seer", "snitch", "spy", "tracker", "trapper", "sheriff", "veteran", "vigilante", "altruist", "medic", "engineer", "mayor", "medium", "swapper", "transporter", "aurial", "hunter", "imitator", "oracle", "vampire hunter", "time lord"],
    "NEUTRAL": ["neutral", "amnesiac", "guardian angel", "survivor", "executioner", "jester", "phantom", "arsonist", "plaguebearer", "the glitch", "werewolf", "doomsayer", "juggernaut"],
    "IMPOSTOR": ["impostor", "grenadier", "morphling", "swooper", "traitor", "blackmailer", "janitor", "miner", "undertaker", "bomber", "escapist", "venerer", "warlock", "poisoner", "underdog"]
}

# Frequency of roles (individually and generally as impostor, neutral, crewmate) on discussion round basis
role_counter = {}
for dr in transcriptions.keys():
    try:
        role_counter[transcriptions[dr]['role']] += 1
    except:
        role_counter[transcriptions[dr]['role']] = 1

role_counter = dict(sorted(role_counter.items(), key=lambda item: item[1]))

# Create a color map based on Imposter, crewmate, neutral
color_map = {}
for r in ROLES_ToR["CREWMATE"]:
    color_map[r] = '#90EE90'  # Light green
for r in ROLES_ToU["CREWMATE"]:
    color_map[r] = '#90EE90'  # Light green
for r in ROLES_ToR["NEUTRAL"]:
    color_map[r] = '#D3D3D3'  # Light gray
for r in ROLES_ToU["NEUTRAL"]:
    color_map[r] = '#D3D3D3'  # Light gray
for r in ROLES_ToR["IMPOSTOR"]:
    color_map[r] = '#FFA07A'  # Light red
for r in ROLES_ToU["IMPOSTOR"]:
    color_map[r] = '#FFA07A'  # Light red

# Extracting keys and values
keys = list(role_counter.keys())
values = list(role_counter.values())

# Creating the bar plot for individual roles
plt.figure(figsize=(10, 6))
bars = plt.bar(keys, values, color=[color_map[key] for key in keys], edgecolor='black')

# Rotating the x-axis labels
plt.xticks(rotation=90)

# Adding labels and title
plt.xlabel('Role')
plt.ylabel('Count')
# plt.title('Frequency of Individual Roles in Extracted Discussion Rounds')

# Creating a legend
crew_patch = mpatches.Patch(color='#90EE90', label='Crewmate')
neutral_patch = mpatches.Patch(color='#D3D3D3', label='Neutral')
impostor_patch = mpatches.Patch(color='#FFA07A', label='Impostor')
plt.legend(handles=[crew_patch, neutral_patch, impostor_patch], title='Role Types', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save plot with high dpi
plt.tight_layout()
plt.savefig(f'{save_dir}frequency_individual_roles.png', dpi=300, bbox_inches='tight')
plt.show()

# Frequency of general roles (Impostor, Neutral, Crewmate)
keys = ['Crewmate', 'Neutral', 'Impostor']
values = [0, 0, 0]
for dr in transcriptions.keys():
    role = transcriptions[dr]['role']
    found = False
    for k in ROLES_ToR.keys():
        if role in ROLES_ToR[k]:
            if k == 'CREWMATE':
                values[0] += 1
            elif k == 'NEUTRAL':
                values[1] += 1
            elif k == 'IMPOSTOR':
                values[2] += 1
            found = True
            break

    if found:
        continue

    for k in ROLES_ToU.keys():
        if role in ROLES_ToU[k]:
            if k == 'CREWMATE':
                values[0] += 1
            elif k == 'NEUTRAL':
                values[1] += 1
            elif k == 'IMPOSTOR':
                values[2] += 1
            break

# Creating the bar plot for general roles
plt.figure(figsize=(10, 6))
bars = plt.bar(keys, values, color=['#90EE90', '#D3D3D3', '#FFA07A'], edgecolor='black')

# Adding labels and title
plt.xlabel('Role')
plt.ylabel('Count')
# plt.title('Frequency of General Roles in Extracted Discussion Rounds')

# Save plot with high dpi
plt.tight_layout()
plt.savefig(f'{save_dir}frequency_general_roles.png', dpi=300, bbox_inches='tight')
plt.show()

# Number and length of utterances for three precision levels
precision_level = ['low_acc', 'mid_acc', 'high_acc']

for pL in precision_level:

    # Number of utterances for different roles
    role_utterance_counter = {}
    for dr in transcriptions.keys():
        try:
            role_utterance_counter[transcriptions[dr]['role']].append(len(transcriptions[dr][pL]))
        except:
            role_utterance_counter[transcriptions[dr]['role']] = [len(transcriptions[dr][pL])]

    keys = list(role_counter.keys())  # Use the same order as previous plots
    values = [np.mean(role_utterance_counter[k]) for k in keys]

    # Creating the bar plot for number of utterances
    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values, color='#ADD8E6', edgecolor='black')  # Light blue

    # Adding labels and title
    plt.xlabel('Role')
    plt.ylabel('Average Number of Utterances')
    # plt.title(f'Average Number of Utterances per Discussion Round ({pL})')

    plt.xticks(rotation=90)

    # Save plot with high dpi
    plt.tight_layout()
    plt.savefig(f'{save_dir}avg_number_utterances_{pL}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Average length of utterances for different roles
    role_utterance_length_counter = {}
    for dr in transcriptions.keys():
        try:
            role_utterance_length_counter[transcriptions[dr]['role']].extend(len(x.split(' ')) for x in transcriptions[dr][pL])
        except:
            role_utterance_length_counter[transcriptions[dr]['role']] = [len(x.split(' ')) for x in transcriptions[dr][pL]]

    values = [np.mean(role_utterance_length_counter[k]) for k in keys]

    # Creating the bar plot for length of utterances
    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values, color='#ADD8E6', edgecolor='black')  # Light blue

    # Adding labels and title
    plt.xlabel('Role')
    plt.ylabel('Average Length of Utterance')
    # plt.title(f'Average Length of Utterances per Role ({pL})')

    plt.xticks(rotation=90)

    # Save plot with high dpi
    plt.tight_layout()
    plt.savefig(f'{save_dir}avg_length_utterances_{pL}.png', dpi=300, bbox_inches='tight')
    plt.show()
