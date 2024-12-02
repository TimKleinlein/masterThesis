from transformers import pipeline
import torch
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


# Check if GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1

# Load the sentiment analysis pipeline with the appropriate device
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)

with open('/kaggle/input/transcriptions-final/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

data = {}
for dr in transcriptions.keys():
    for ind, utt in enumerate(transcriptions[dr]['high_acc']):
        data[f'{dr}_{ind}'] = {'text': utt, 'label': transcriptions[dr]['role']}

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

for utt in data.keys():
    found = False
    for r in ROLES_ToR.keys():
        for sr in ROLES_ToR[r]:
            if sr == data[utt]['label']:
                data[utt]['label'] = r
                found = True
                break
    if not found:
        for r in ROLES_ToU.keys():
            for sr in ROLES_ToU[r]:
                if sr == data[utt]['label']:
                    data[utt]['label'] = r
                    found = True
                    break

utt_impostors = []
utt_crewmates = []
utt_neutrals = []
for k in data.keys():
    if data[k]['label'] == 'IMPOSTOR':
        utt_impostors.append(data[k]['text'])
    elif data[k]['label'] == 'CREWMATE':
        utt_crewmates.append(data[k]['text'])
    elif data[k]['label'] == 'NEUTRAL':
        utt_neutrals.append(data[k]['text'])

labels_dic = {}
scores_dic = {}

for ind, utterances in enumerate([utt_crewmates, utt_neutrals, utt_impostors]):
    labels_dic[ind] = []
    scores_dic[ind] = []

    # Define a mapping from labels to numerical values
    label_mapping = {"negative": -1, "neutral": 0, "positive": 1}

    # Perform sentiment analysis on the first 10 utterances
    results = sentiment_pipeline(utterances, return_all_scores=True)

    # Process and display the results with numerical labels
    for i, result in enumerate(results):
        pred_label = result[[x['score'] for x in result].index(max([x['score'] for x in result]))]['label']

        overall_score = 0
        for label_score in result:
            label = label_score['label']
            score = label_score['score']
            overall_score += label_mapping[label] * score
        labels_dic[ind].append(pred_label)
        scores_dic[ind].append(overall_score)


scores = {}
neg = 0
neu = 0
pos = 0
for l in labels_dic[0]:
    if l == 'negative':
        neg += 1
    elif l == 'neutral':
        neu +=1
    elif l == 'positive':
        pos += 1
print(f'{neg} {neu} {pos}')
total = neg + neu + pos
print(f'{neg / total} {neu / total} {pos / total}')
scores['Crewmates'] = {'Negative': neg / total, 'Neutral': neu / total, 'Positive': pos / total}

neg = 0
neu = 0
pos = 0
for l in labels_dic[1]:
    if l == 'negative':
        neg += 1
    elif l == 'neutral':
        neu +=1
    elif l == 'positive':
        pos += 1
print(f'{neg} {neu} {pos}')
total = neg + neu + pos
print(f'{neg / total} {neu / total} {pos / total}')
scores['Neutrals'] = {'Negative': neg / total, 'Neutral': neu / total, 'Positive': pos / total}

neg = 0
neu = 0
pos = 0
for l in labels_dic[2]:
    if l == 'negative':
        neg += 1
    elif l == 'neutral':
        neu +=1
    elif l == 'positive':
        pos += 1
print(f'{neg} {neu} {pos}')
total = neg + neu + pos
print(f'{neg / total} {neu / total} {pos / total}')
scores['Impostors'] = {'Negative': neg / total, 'Neutral': neu / total, 'Positive': pos / total}

# create bar chart

categories = ["Negative", "Neutral", "Positive"]
crew_scores = [scores["Crewmates"][cat] for cat in categories]
neutral_scores = [scores["Neutrals"][cat] for cat in categories]
impostor_scores = [scores["Impostors"][cat] for cat in categories]

# Number of categories
n_categories = len(categories)

# Bar width
bar_width = 0.2

# Bar positions
ind = np.arange(n_categories)

# Set the style to match the other scripts
plt.style.use('seaborn-deep')

# General settings for plots
plt.rcParams.update({
    'figure.figsize': (10, 6),
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

# Plotting the grouped bar chart
fig, ax = plt.subplots()

# Define lighter colors
light_green = '#90EE90'  # Light Green
light_gray = '#D3D3D3'   # Light Gray (for light black)
light_red = '#FFA07A'    # Light Red

# Create bars
p1 = ax.bar(ind - bar_width, crew_scores, bar_width, label='Crewmates', color=light_green, edgecolor='black')
p2 = ax.bar(ind, neutral_scores, bar_width, label='Neutrals', color=light_gray, edgecolor='black')
p3 = ax.bar(ind + bar_width, impostor_scores, bar_width, label='Impostors', color=light_red, edgecolor='black')

# Adding labels and title
ax.set_xlabel('Sentiment')
ax.set_ylabel('Ratio')
#ax.set_title('Ratio of Sentiment Labels by Role')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

# Save the plot with high dpi
save_dir = '/kaggle/working/'
plt.tight_layout()
plt.savefig(f'{save_dir}sentiment_ratio_by_role_lighter.png', dpi=300, bbox_inches='tight')

# Display the chart
plt.show()



# plot mean score
groups = list(scores_dic.keys())
values = [np.mean(scores_dic[x]) for x in list(scores_dic.keys())]

# Set the style to match the other scripts
plt.style.use('seaborn-deep')

# General settings for plots
plt.rcParams.update({
    'figure.figsize': (10, 6),
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

# Plotting the simple bar chart
fig, ax = plt.subplots()

# Define lighter colors
light_green = '#90EE90'  # Light Green
light_gray = '#D3D3D3'   # Light Gray (for light black)
light_red = '#FFA07A'    # Light Red

# Assigning colors based on roles
colors = [light_green, light_gray, light_red]

# Create bars with edge color
ax.bar(groups, values, color=colors, edgecolor='black')

# Adding labels and title
ax.set_xlabel('Roles')
ax.set_ylabel('Mean score of utterances')
#ax.set_title('Mean Sentiment Scores of Utterances by Role')

# Setting y-axis range
ax.set_ylim(-0.2, .2)

# Save the plot with high dpi
save_dir = '/kaggle/working/'
plt.tight_layout()
plt.savefig(f'{save_dir}mean_sentiment_scores_by_role.png', dpi=300, bbox_inches='tight')

# Display the chart
plt.show()

