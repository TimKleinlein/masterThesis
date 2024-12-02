import os
import pickle as pkl
import math
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import stopwords, brown
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from wordcloud import WordCloud
from matplotlib import pyplot as plt

# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('brown')

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

# Create the output directory if it doesn't exist
output_dir = 'data/word_frequency'
os.makedirs(output_dir, exist_ok=True)

# Load transcriptions
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# Step 1: Text Preprocessing
text = ''
for k in transcriptions.keys():
    for utt in transcriptions[k]['low_acc']:
        text = text + ' ' + utt

# Tokenize text by words
words = word_tokenize(text)

# Create an empty list to store words without punctuation
words_no_punc = [word.lower() for word in words if word.isalpha()]

# List of stopwords
stopwords_list = stopwords.words("english")
stopwords_list2 = []

with open('data/word_frequency/stop_words_english.txt', 'r') as file:
    for line in file:
        stopwords_list2.append(line.strip())

# Create an empty list to store clean words
clean_words = [word for word in words_no_punc if word not in stopwords_list and word not in stopwords_list2]

# Step 2: Word Frequency Analysis
fdist = FreqDist(clean_words)

plt.figure()
fdist.plot(20)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_20_words_overall.png'), dpi=300, bbox_inches='tight')
plt.show()

# Convert word list to a single string
clean_words_string = " ".join(clean_words)

# Generating the word cloud and saving it
wordcloud = WordCloud(background_color="white").generate(clean_words_string)
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wordcloud_overall.png'), dpi=300, bbox_inches='tight')
plt.show()

# Step 3: TF-IDF Analysis
tf = defaultdict(int)
for word in clean_words:
    tf[word] += 1

# Function to calculate IDF for a single word
def calculate_idf(word):
    num_docs_containing_word = sum(1 for doc in brown.fileids() if word in brown.words(doc))
    return math.log((len(brown.fileids()) + 1) / (num_docs_containing_word + 1))

# Calculate IDF using Brown corpus
idf = defaultdict(lambda: 0)
unique_words = set(clean_words)

for word in tqdm(unique_words, desc="Calculating IDF"):
    idf[word] = calculate_idf(word)

# Calculate TF-IDF
tf_idf = {word: tf[word] * idf[word] for word in clean_words}

# Sort words by TF-IDF score
sorted_tf_idf = sorted(tf_idf.items(), key=lambda item: item[1], reverse=True)

# Print the top 20 words with the highest TF-IDF scores
print("Top 20 unusual words in the text:")
for word, score in sorted_tf_idf[:20]:
    print(f"{word}: {score}")

# Save dictionary with tf_idf scores
with open(os.path.join(output_dir, 'tf_idf_overall.pkl'), 'wb') as f:
    pkl.dump(sorted_tf_idf, f)

# Save the word cloud for the top 100 words with highest TF-IDF scores
top_100_words = dict(sorted_tf_idf[:100])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(top_100_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wordcloud_cfa_overall.png'), dpi=300, bbox_inches='tight')
plt.show()

# Step 4: Word Frequency Analysis for 'ozzaworld'
# Find streamer with most utterances
streamer_dic = {}
for dr in transcriptions.keys():
    s = dr.split('_')[4]
    streamer_dic[s] = streamer_dic.get(s, 0) + len(transcriptions[dr]['low_acc'])

# Create text of utterances of ozzaworld as CREWMATE and IMPOSTOR
text_crewmate = ''
text_impostor = ''

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

for utt in transcriptions.keys():
    found = False
    for r in ROLES_ToR.keys():
        for sr in ROLES_ToR[r]:
            if sr == transcriptions[utt]['role']:
                transcriptions[utt]['role'] = r
                found = True
                break
    if not found:
        for r in ROLES_ToU.keys():
            for sr in ROLES_ToU[r]:
                if sr == transcriptions[utt]['role']:
                    transcriptions[utt]['role'] = r
                    found = True
                    break

# Remove neutral roles from data
data = {k: v for k, v in transcriptions.items() if v['role'] != 'NEUTRAL'}

# Get only ozzaworld entries
data = {k: v for k, v in data.items() if k.split('_')[4] == 'ozzaworld'}

for k in data.keys():
    if data[k]['role'] == 'CREWMATE':
        for utt in data[k]['low_acc']:
            text_crewmate = text_crewmate + ' ' + utt
    elif data[k]['role'] == 'IMPOSTOR':
        for utt in data[k]['low_acc']:
            text_impostor = text_impostor + ' ' + utt

fdist_dic = {}

# Analyze text for each role (CREWMATE and IMPOSTOR)
for ind, (role_text, role_name) in enumerate([(text_crewmate, 'CREWMATE'), (text_impostor, 'IMPOSTOR')]):
    # Tokenize text by words
    words = word_tokenize(role_text)

    # Create an empty list to store words
    words_no_punc = [word.lower() for word in words if word.isalpha()]

    # Create an empty list to store clean words
    clean_words = [word for word in words_no_punc if word not in stopwords_list and word not in stopwords_list2]

    # Find the frequency of words and save the plot
    fdist = FreqDist(clean_words)
    fdist_dic[role_name] = fdist

    # Plot and save the top 20 words
    plt.figure()
    fdist.plot(20)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ozzaworld_top_20_words_{role_name}.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Normalize the word frequencies for IMPOSTOR to compare with CREWMATE
for k in fdist_dic['IMPOSTOR'].keys():
    fdist_dic['IMPOSTOR'][k] = fdist_dic['IMPOSTOR'][k] * (12111 / 4843)

# Create pairwise distances
pairwise_distances = {}
for word in fdist_dic['CREWMATE'].keys():
    crew_freq = fdist_dic['CREWMATE'][word]
    imp_freq = fdist_dic['IMPOSTOR'].get(word, 0)
    pairwise_distances[word] = crew_freq - imp_freq

# Add words that are only said in IMPOSTOR setting
for word in fdist_dic['IMPOSTOR'].keys():
    if word not in fdist_dic['CREWMATE']:
        pairwise_distances[word] = -fdist_dic['IMPOSTOR'][word]

sorted_distances = sorted(pairwise_distances.items(), key=lambda item: item[1])

# Get the 20 words with the lowest scores
lowest_20 = sorted_distances[:20]

# Get the 20 words with the highest scores
highest_20 = sorted_distances[-20:]

print("20 words with the lowest scores:")
for word, score in lowest_20:
    print(f"{word}: {round(score, 2)}")

print("\n20 words with the highest scores:")
for word, score in highest_20:
    print(f"{word}: {round(score, 2)}")

# Plot and save the lowest 20 scores
lowest_20_words, lowest_20_scores = zip(*lowest_20)
plt.figure()
plt.bar(lowest_20_words, lowest_20_scores, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Difference in Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ozzaworld_lowest_20_word_frequencies.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot and save the highest 20 scores
highest_20_words, highest_20_scores = zip(*highest_20)
plt.figure()
plt.bar(highest_20_words, highest_20_scores, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Difference in Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ozzaworld_highest_20_word_frequencies.png'), dpi=300, bbox_inches='tight')
plt.show()

# Step 5: TF-IDF Analysis for 'ozzaworld'
def compute_tf_idf(text):
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform([text])
    tf_idf_scores = dict(zip(vectorizer.get_feature_names_out(), tf_idf_matrix.toarray().flatten()))
    return tf_idf_scores

# Compute TF-IDF scores for CREWMATE and IMPOSTOR texts
tf_idf_crewmate = compute_tf_idf(text_crewmate)
tf_idf_impostor = compute_tf_idf(text_impostor)

# Normalize TF-IDF scores for IMPOSTOR
normalization_factor = 12111 / 4843
tf_idf_impostor_normalized = {word: score * normalization_factor for word, score in tf_idf_impostor.items()}

# Compare TF-IDF scores between CREWMATE and IMPOSTOR
pairwise_distances_tf_idf = {}
for word, crew_freq in tf_idf_crewmate.items():
    imp_freq = tf_idf_impostor_normalized.get(word, 0)
    pairwise_distances_tf_idf[word] = crew_freq - imp_freq

# Add words that are only in IMPOSTOR
for word, imp_freq in tf_idf_impostor_normalized.items():
    if word not in tf_idf_crewmate:
        pairwise_distances_tf_idf[word] = -imp_freq

sorted_distances_tf_idf = sorted(pairwise_distances_tf_idf.items(), key=lambda item: item[1])

# Get the 20 words with the lowest and highest scores
lowest_20_tf_idf = sorted_distances_tf_idf[:20]
highest_20_tf_idf = sorted_distances_tf_idf[-20:]

# Print the results
print("20 words with the lowest TF-IDF scores:")
for word, score in lowest_20_tf_idf:
    print(f"{word}: {round(score, 2)}")

print("\n20 words with the highest TF-IDF scores:")
for word, score in highest_20_tf_idf:
    print(f"{word}: {round(score, 2)}")

# Plot and save the lowest 20 TF-IDF scores
lowest_20_words_tf_idf, lowest_20_scores_tf_idf = zip(*lowest_20_tf_idf)
plt.figure()
plt.bar(lowest_20_words_tf_idf, lowest_20_scores_tf_idf, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Difference in TF-IDF Scores")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ozzaworld_lowest_20_tfidf_words.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot and save the highest 20 TF-IDF scores
highest_20_words_tf_idf, highest_20_scores_tf_idf = zip(*highest_20_tf_idf)
plt.figure()
plt.bar(highest_20_words_tf_idf, highest_20_scores_tf_idf, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Difference in TF-IDF Scores")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ozzaworld_highest_20_tfidf_words.png'), dpi=300, bbox_inches='tight')
plt.show()
