import logging
import os
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle as pkl
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import Dataset

# Set the environment variable to use only GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the evaluation_results directory if it doesn't exist
evaluation_results_dir = "data/majority_vote"
os.makedirs(evaluation_results_dir, exist_ok=True)

# Set up logging
logging.basicConfig(filename=os.path.join(evaluation_results_dir, 'results_majority_vote.txt'), level=logging.INFO, format='%(message)s')

with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# Convert roles in transcription to general roles for evaluation purposes
correct_roles = {}
for dr in transcriptions.keys():
    correct_roles[dr] = transcriptions[dr]['role']

ROLES_ToR = {
    "CREWMATE": ["crewmate", "mayor", "medium", "swapper", "time master", "engineer", "sheriff", "deputy", "lighter",
                 "detective", "medic", "seer", "hacker", "tracker", "snitch", "spy", "portalmaker", "security guard",
                 "medium", "trapper", "nice guesser", "bait", "shifter"],
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

for dr in correct_roles.keys():
    found = False
    for r in ROLES_ToR.keys():
        for sr in ROLES_ToR[r]:
            if sr == correct_roles[dr]:
                correct_roles[dr] = r
                found = True
                break
    if not found:
        for r in ROLES_ToU.keys():
            for sr in ROLES_ToU[r]:
                if sr == correct_roles[dr]:
                    correct_roles[dr] = r
                    found = True
                    break

# Remove neutral roles from data
correct_roles = {k: v for k, v in correct_roles.items() if v != 'NEUTRAL'}

# Load the tokenizer
# This will be dynamic, based on each fold
# Load the model
# This will also be dynamic, based on each fold

# Evaluation for each fold
accuracy_modes = ["low_acc", "mid_acc", "high_acc"]
num_folds = 5

for acc_mode in accuracy_modes:
    for fold in range(1, num_folds + 1):
        logging.info(f"Evaluating fold {fold} for accuracy mode: {acc_mode}")

        # Load test indices
        test_indices_path = os.path.join("data/bert_classifier", acc_mode, f"test_indices_fold_{fold}.json")
        with open(test_indices_path, 'r') as f:
            test_indices = json.load(f)

        # Prepare test dataset
        data = {}
        for dr in transcriptions.keys():
            for ind, utt in enumerate(transcriptions[dr][acc_mode]):
                data[f'{dr}_{ind}'] = {'text': utt, 'label': transcriptions[dr]['role'], 'group': dr}

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

        # Remove neutral roles from data
        data = {k: v for k, v in data.items() if v['label'] != 'NEUTRAL'}

        texts = [data[key]['text'] for key in data]
        labels = [data[key]['label'] for key in data]
        groups = [data[key]['group'] for key in data]
        labels = [0 if x == 'CREWMATE' else 1 if x == 'IMPOSTOR' else 2 for x in labels]

        # Create a Dataset from the dictionary
        data = {'text': texts, 'label': labels, 'group': groups}
        dataset = Dataset.from_dict(data)

        # Filter test dataset using test indices
        test_dataset = dataset.select(test_indices)

        # Load the corresponding tokenizer and model for the fold
        tokenizer_path = os.path.join("data/bert_classifier", acc_mode, f"best_model_fold_{fold}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(tokenizer_path)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        def preprocess(text):
            return tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

        # Use majority vote to predict discussion rounds (if draw -> predict CREWMATE)
        evaluation_dic = {}
        test_groups = set(test_dataset['group'])
        for dr in test_groups:
            CREWMATE_count = 0
            IMPOSTOR_count = 0
            for utt in transcriptions[dr][acc_mode]:
                input_text = utt
                encoded_input = preprocess(input_text)
                encoded_input = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in
                                 encoded_input.items()}

                with torch.no_grad():
                    outputs = model(**encoded_input)
                    logits = outputs.logits

                # Convert logits to probabilities
                probabilities = torch.softmax(logits, dim=1)

                # Get the predicted class
                predicted_class = torch.argmax(probabilities, dim=1).item()

                # Increase counter of predicted class
                if predicted_class == 0:
                    CREWMATE_count += 1
                elif predicted_class == 1:
                    IMPOSTOR_count += 1

            if len(transcriptions[dr][acc_mode]) != 0:
                if IMPOSTOR_count > CREWMATE_count:
                    predicted_role = 'IMPOSTOR'
                else:
                    predicted_role = 'CREWMATE'
                evaluation_dic[dr] = {'number_utterances': len(transcriptions[dr][acc_mode]),
                                      'predicted_role': predicted_role}

        # EVALUATION
        # Compare predictions with majority vote with true identity of each discussion round

        # Extract predicted and actual roles
        predicted_roles = [evaluation_dic[dr]['predicted_role'] for dr in evaluation_dic.keys()]
        actual_roles = [correct_roles[dr] for dr in evaluation_dic.keys()]

        # Calculate accuracy
        accuracy = accuracy_score(actual_roles, predicted_roles)
        print(f'Accuracy: {accuracy}')
        logging.info(f'Accuracy: {accuracy}')

        # Create confusion matrix
        conf_matrix = confusion_matrix(actual_roles, predicted_roles, labels=["CREWMATE", "IMPOSTOR"])

        # Log confusion matrix
        logging.info('Confusion Matrix:')
        logging.info(conf_matrix)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["CREWMATE", "IMPOSTOR"], yticklabels=["CREWMATE", "IMPOSTOR"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save confusion matrix plot
        plt.savefig(os.path.join(evaluation_results_dir, f'confusion_matrix_majority_vote_{acc_mode}_fold_{fold}.png'))

        # Display the plot
        plt.show()

        # EVALUATION for discussion rounds with at least 3 utterances
        filtered_evaluation_dic = {dr: eval_dict for dr, eval_dict in evaluation_dic.items() if eval_dict['number_utterances'] >= 3}

        # Extract predicted and actual roles for filtered discussion rounds
        filtered_predicted_roles = [filtered_evaluation_dic[dr]['predicted_role'] for dr in filtered_evaluation_dic.keys()]
        filtered_actual_roles = [correct_roles[dr] for dr in filtered_evaluation_dic.keys()]

        # Calculate accuracy for filtered discussion rounds
        filtered_accuracy = accuracy_score(filtered_actual_roles, filtered_predicted_roles)
        print(f'Filtered Accuracy (>=3 utterances): {filtered_accuracy}')
        logging.info(f'Filtered Accuracy (>=3 utterances): {filtered_accuracy}')

        # Create confusion matrix for filtered discussion rounds
        filtered_conf_matrix = confusion_matrix(filtered_actual_roles, filtered_predicted_roles, labels=["CREWMATE", "IMPOSTOR"])

        # Log filtered confusion matrix
        logging.info('Filtered Confusion Matrix (>=3 utterances):')
        logging.info(filtered_conf_matrix)

        # Plot filtered confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(filtered_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["CREWMATE", "IMPOSTOR"], yticklabels=["CREWMATE", "IMPOSTOR"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Filtered Confusion Matrix (>=3 utterances)')

        # Save filtered confusion matrix plot
        plt.savefig(os.path.join(evaluation_results_dir, f'filtered_confusion_matrix_majority_vote_{acc_mode}_fold_{fold}.png'))

        # Display the plot
        plt.show()
