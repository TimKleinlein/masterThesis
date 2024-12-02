import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.manifold import MDS
import pickle as pkl
import warnings


with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

texts = []
for dr in transcriptions.keys():
    for utt in transcriptions[dr]['low_acc']:
        texts.append(utt)


def preprocess_data(documents):
    stop_words = stopwords.words('english')

    # Tokenize and remove stopwords
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]

    return texts

processed_texts = preprocess_data(texts)

# Create Dictionary
id2word = corpora.Dictionary(processed_texts)

# Create Corpus
texts = processed_texts

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Set number of topics
num_topics = 3

# Build LDA model
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10, alpha='auto', per_word_topics=True)

# Print the keywords for each topic
pprint(lda_model.print_topics())

coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)



# Function to plot the distance between topics
def plot_topic_distances(lda_model, num_topics, save_dir):
    # Extract topic-word distributions and create a distance matrix
    topics = lda_model.get_topics()
    distance_matrix = np.zeros((num_topics, num_topics))

    for i in range(num_topics):
        for j in range(num_topics):
            distance_matrix[i, j] = np.linalg.norm(topics[i] - topics[j])

    # Use MDS to reduce the dimensionality to 2D for visualization
    mds = MDS(n_components=2, dissimilarity='precomputed')
    topic_positions = mds.fit_transform(distance_matrix)

    # Suppress the UserWarning related to edgecolor/facecolor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

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

        plt.figure(figsize=(10, 6))
        plt.scatter(topic_positions[:, 0], topic_positions[:, 1], c='blue', edgecolor='black', s=100)

        # Positioning Topic 1 and Topic 2 on the right
        plt.text(topic_positions[0, 0] + 0.01, topic_positions[0, 1], 'Topic 1', fontsize=12, ha='left', va='center')
        plt.text(topic_positions[1, 0] + 0.01, topic_positions[1, 1], 'Topic 2', fontsize=12, ha='left', va='center')

        # Positioning Topic 3 on the left
        plt.text(topic_positions[2, 0] - 0.01, topic_positions[2, 1], 'Topic 3', fontsize=12, ha='right', va='center')

        # plt.title('Topic Distances')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.tight_layout()

        # Save the plot with high dpi
        plt.savefig(f'{save_dir}topic_distances.png', dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()


# Function to plot word clouds for each topic
def plot_word_clouds(lda_model, num_topics, save_dir, num_words=10):
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)

    fig, axes = plt.subplots(1, num_topics, figsize=(15, 8), sharex=True, sharey=True)

    # Set the style to match the other scripts
    plt.style.use('seaborn-deep')

    # General settings for plots
    plt.rcParams.update({
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

    for i, ax in enumerate(axes.flatten()):
        topic_words = dict(topics[i][1])
        wordcloud = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(topic_words)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Topic {i + 1}', fontsize=12, fontfamily='Times New Roman')

        # Save each word cloud with high dpi
        plt.savefig(f'{save_dir}wordcloud_topic_{i + 1}.png', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


# Define save path for the plot
save_dir = '/kaggle/working/'

# Plotting the topic distances and saving the plot
plot_topic_distances(lda_model, num_topics, save_dir)

# Plotting the word clouds for each topic and saving them
plot_word_clouds(lda_model, num_topics, save_dir)

