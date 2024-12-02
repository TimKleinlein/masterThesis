import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle as pkl
import matplotlib.pyplot as plt
from collections import Counter

# Load transcription data
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# Organize the data by utterance and label
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

# Convert roles to general categories
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

# Separate utterances by role
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

# POS tagging function
def pos_tagging_nltk(utterances):
    pos_counts = []
    for utterance in utterances:
        for sent in sent_tokenize(utterance):
            wordtokens = word_tokenize(sent)
            words = nltk.word_tokenize(utterance)
            pos_tags = nltk.pos_tag(words)
            pos_count = Counter(tag for word, tag in pos_tags)
            pos_counts.append(pos_count)
    return pos_counts

# Run POS tagging for each role
counts_crewmate = pos_tagging_nltk(utt_crewmates)
counts_neutral = pos_tagging_nltk(utt_neutrals)
counts_impostor = pos_tagging_nltk(utt_impostors)

# Aggregate POS counts for each role
def aggregate_pos_counts(counts):
    pos_counter = {}
    for utt in counts:
        for pos in utt.keys():
            pos_counter[pos] = pos_counter.get(pos, 0) + utt[pos]
    return pos_counter

pos_counter_crewmate = aggregate_pos_counts(counts_crewmate)
pos_counter_neutral = aggregate_pos_counts(counts_neutral)
pos_counter_impostor = aggregate_pos_counts(counts_impostor)

role2ind = {0: 'Crewmate', 1: 'Neutral', 2: 'Impostor'}

# Set the style to match the other scripts
plt.style.use('seaborn-deep')

# General settings for plots
plt.rcParams.update({
    'figure.figsize': (12, 8),
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

# Save directory
save_dir = '/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/'

# Plotting POS frequencies for each role
for ind, pos_dic in enumerate([pos_counter_crewmate, pos_counter_neutral, pos_counter_impostor]):

    for pos in ['.', ',', ':', "''"]:
        del (pos_dic[pos])
    # Calculate total number of POS tags
    total_pos = sum(pos_dic.values())

    # Calculate the threshold for 1% of the total
    threshold = 0.01 * total_pos

    # Separate the main POS tags and the rest
    main_pos = {tag: count for tag, count in pos_dic.items() if count >= threshold}
    rest_pos = {tag: count for tag, count in pos_dic.items() if count < threshold}

    # Calculate the 'rest' category
    rest_sum = sum(rest_pos.values())

    # Sort the main POS tags by frequency
    sorted_pos = sorted(main_pos.items(), key=lambda item: item[1], reverse=True)

    # Add the 'rest' category to the end
    sorted_pos.append(('Rest', rest_sum))

    # Extract keys and values for plotting
    tags, counts = zip(*sorted_pos)
    counts = tuple([x / total_pos for x in list(counts)])

    # Plotting the barplot
    plt.figure(figsize=(12, 8))
    plt.bar(tags, counts, color='skyblue', edgecolor='black')
    plt.xlabel('POS Tags')
    plt.ylabel('Frequency')
    plt.title(f'POS Tag Frequencies {role2ind[ind]}')
    plt.xticks(rotation=90)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_dir}pos_frequencies_{role2ind[ind].lower()}.png', dpi=300, bbox_inches='tight')

    plt.show()

# Function to calculate ratios
def calculate_ratios(pos_counter):
    total = sum(pos_counter.values())
    return {tag: count / total for tag, count in pos_counter.items()}

ratios_crewmate = calculate_ratios(pos_counter_crewmate)
ratios_neutral = calculate_ratios(pos_counter_neutral)
ratios_impostor = calculate_ratios(pos_counter_impostor)

# Function to calculate differences
def calculate_difference(ratios1, ratios2):
    all_tags = set(ratios1.keys()).union(ratios2.keys())
    differences = {tag: ratios1.get(tag, 0) - ratios2.get(tag, 0) for tag in all_tags}
    return differences

# Differences between the classes
diff_crewmate_neutral = calculate_difference(ratios_crewmate, ratios_neutral)
diff_crewmate_impostor = calculate_difference(ratios_crewmate, ratios_impostor)
diff_neutral_impostor = calculate_difference(ratios_neutral, ratios_impostor)

# Plotting function for differences
def plot_differences(differences, title, filename):
    sorted_diff = sorted(differences.items(), key=lambda item: item[1], reverse=True)
    tags, counts = zip(*sorted_diff)

    plt.figure(figsize=(12, 8))
    plt.bar(tags, counts, color='skyblue', edgecolor='black')
    plt.xlabel('POS Tags')
    plt.ylabel('Difference in Ratios')
    plt.title(title)
    plt.xticks(rotation=90)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_dir}{filename}.png', dpi=300, bbox_inches='tight')

    plt.show()

# Plot differences and save the plots
plot_differences(diff_crewmate_neutral, 'Difference in POS Tag Ratios: Crewmate vs Neutral', 'diff_crewmate_neutral')
plot_differences(diff_crewmate_impostor, 'Difference in POS Tag Ratios: Crewmate vs Impostor', 'diff_crewmate_impostor')
plot_differences(diff_neutral_impostor, 'Difference in POS Tag Ratios: Neutral vs Impostor', 'diff_neutral_impostor')






# now for single streamers


# Define the save directory
save_dir = '/kaggle/working'

# Set the style to match the other scripts
plt.style.use('seaborn-deep')

# General settings for plots
plt.rcParams.update({
    'figure.figsize': (12, 8),
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

# POS tagging function
def pos_tagging_nltk(utterances):
    pos_counts = []
    for utterance in utterances:
        for sent in sent_tokenize(utterance):
            words = word_tokenize(sent)
            pos_tags = nltk.pos_tag(words)
            pos_count = Counter(tag for word, tag in pos_tags)
            pos_counts.append(pos_count)
    return pos_counts

# Function to aggregate POS counts
def aggregate_pos_counts(counts):
    pos_counter = {}
    for utt in counts:
        for pos in utt.keys():
            pos_counter[pos] = pos_counter.get(pos, 0) + utt[pos]
    return pos_counter

# Function to calculate ratios
def calculate_ratios(pos_counter):
    total = sum(pos_counter.values())
    return {tag: count / total for tag, count in pos_counter.items()}

# Function to calculate differences
def calculate_difference(ratios1, ratios2):
    all_tags = set(ratios1.keys()).union(ratios2.keys())
    differences = {tag: ratios1.get(tag, 0) - ratios2.get(tag, 0) for tag in all_tags}
    return differences

# Function to plot differences
def plot_differences(differences, title, filename):
    sorted_diff = sorted(differences.items(), key=lambda item: item[1], reverse=True)
    tags, counts = zip(*sorted_diff)

    plt.figure(figsize=(12, 8))
    plt.bar(tags, counts, color='skyblue', edgecolor='black')
    plt.xlabel('POS Tags')
    plt.ylabel('Difference in Ratios')
    plt.title(title)
    plt.xticks(rotation=90)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_dir}{filename}.png', dpi=300, bbox_inches='tight')

    plt.show()

# Main analysis for each streamer
for streamer in ['ozzaworld', 'zeroyalviking']:

    data = {}
    for dr in transcriptions.keys():
        if dr.split('_')[-1] != streamer:
            continue
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

    # Convert roles to general categories
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

    # Separate utterances by role
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

    # Run POS tagging
    counts_crewmate = pos_tagging_nltk(utt_crewmates)
    counts_neutral = pos_tagging_nltk(utt_neutrals)
    counts_impostor = pos_tagging_nltk(utt_impostors)

    # Aggregate POS counts
    pos_counter_crewmate = aggregate_pos_counts(counts_crewmate)
    pos_counter_neutral = aggregate_pos_counts(counts_neutral)
    pos_counter_impostor = aggregate_pos_counts(counts_impostor)

    role2ind = {0: 'Crewmate', 1: 'Neutral', 2: 'Impostor'}

    # Plot POS frequencies for each role
    for ind, pos_dic in enumerate([pos_counter_crewmate, pos_counter_neutral, pos_counter_impostor]):

        for pos in ['.', ',', ':', "''"]:
            if pos in pos_dic.keys():
                del(pos_dic[pos])
        # Calculate total number of POS tags
        total_pos = sum(pos_dic.values())

        # Calculate the threshold for 1% of the total
        threshold = 0.01 * total_pos

        # Separate the main POS tags and the rest
        main_pos = {tag: count for tag, count in pos_dic.items() if count >= threshold}
        rest_pos = {tag: count for tag, count in pos_dic.items() if count < threshold}

        # Calculate the 'rest' category
        rest_sum = sum(rest_pos.values())

        # Sort the main POS tags by frequency
        sorted_pos = sorted(main_pos.items(), key=lambda item: item[1], reverse=True)

        # Add the 'rest' category to the end
        sorted_pos.append(('Rest', rest_sum))

        # Extract keys and values for plotting
        tags, counts = zip(*sorted_pos)
        counts = tuple([x / total_pos for x in list(counts)])

        # Plotting the barplot
        plt.figure(figsize=(12, 8))
        plt.bar(tags, counts, color='skyblue', edgecolor='black')
        plt.xlabel('POS Tags')
        plt.ylabel('Frequency')
        plt.title(f'POS Tag Frequencies {role2ind[ind]} ({streamer})')
        plt.xticks(rotation=90)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_dir}pos_frequencies_{role2ind[ind].lower()}_{streamer}.png', dpi=300, bbox_inches='tight')

        plt.show()

    # Calculate POS tag ratios
    ratios_crewmate = calculate_ratios(pos_counter_crewmate)
    ratios_neutral = calculate_ratios(pos_counter_neutral)
    ratios_impostor = calculate_ratios(pos_counter_impostor)

    # Differences between the classes
    diff_crewmate_neutral = calculate_difference(ratios_crewmate, ratios_neutral)
    diff_crewmate_impostor = calculate_difference(ratios_crewmate, ratios_impostor)
    diff_neutral_impostor = calculate_difference(ratios_neutral, ratios_impostor)

    # Plot differences and save the plots
    plot_differences(diff_crewmate_neutral, f'Difference in POS Tag Ratios: Crewmate vs Neutral ({streamer})', f'diff_crewmate_neutral_{streamer}')
    plot_differences(diff_crewmate_impostor, f'Difference in POS Tag Ratios: Crewmate vs Impostor ({streamer})', f'diff_crewmate_impostor_{streamer}')
    plot_differences(diff_neutral_impostor, f'Difference in POS Tag Ratios: Neutral vs Impostor ({streamer})', f'diff_neutral_impostor_{streamer}')

