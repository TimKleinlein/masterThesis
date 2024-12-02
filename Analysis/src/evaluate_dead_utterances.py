import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch
import logging
import os
import pickle as pkl
import json
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set the environment variable to use only GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer for low_acc fold 1
model_path = "data/bert_classifier/low_acc/best_model_fold_1"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load transcriptions data
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

# List of specific discussion rounds to test
dead_discussion_rounds = ['2022-02-16_S1_l10_5_ozzaworld', '2022-05-24_S1_l3_5_courtilly', '2022-05-24_S1_l11_4_irepptar', '2022-05-24_S1_l11_4_jayfletcher88', '2022-02-12_S1_l20_4_courtilly', '2022-05-24_S1_l3_4_irepptar', '2022-02-12_S1_l22_3_ozzaworld', '2022-05-24_S1_l2_3_irepptar', '2022-03-10_S1_l5_3_skadj', '2022-03-03_S1_l2_3_ozzaworld', '2022-02-09_S1_l14_3_ozzaworld']

# Filter and prepare the data for the specific discussion rounds
data = []
for dr in dead_discussion_rounds:
    if dr in transcriptions:
        for utt in transcriptions[dr]['low_acc']:
            data.append({'text': utt, 'label': transcriptions[dr]['role'], 'group': dr})

# Map roles to CREWMATE and IMPOSTOR categories
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

# Relabel the data to CREWMATE and IMPOSTOR categories
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

# Create a Dataset from the dictionary
test_dataset = datasets.Dataset.from_dict({'text': texts, 'label': labels})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the Trainer
trainer = Trainer(
    model=model,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics  # Include accuracy computation
)

# Evaluate the model
eval_results = trainer.evaluate()

# Save the evaluation results
output_dir = "data/evaluate_dead_videos"
os.makedirs(output_dir, exist_ok=True)
eval_results_path = os.path.join(output_dir, "evaluation_results.txt")
with open(eval_results_path, "w") as writer:
    for key, value in eval_results.items():
        writer.write(f"{key}: {value}\n")

# Generate and save confusion matrix
predictions, labels, _ = trainer.predict(tokenized_test_dataset)
predictions = np.argmax(predictions, axis=1)
cm = confusion_matrix(labels, predictions, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CREWMATE', 'IMPOSTOR'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.subplots_adjust(left=0.2)  # Adjust the padding to ensure the y-axis label is not cut off
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"Evaluation results saved to {eval_results_path}")
print(f"Confusion matrix saved to {cm_path}")
