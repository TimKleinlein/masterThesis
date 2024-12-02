import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random
import os
from tqdm import tqdm
from collections import defaultdict

# Set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "data/bert_classifier/low_acc/best_model_fold_1"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Helper function to perform a forward pass
def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    outputs = model(inputs, token_type_ids=token_type_ids,
                    position_ids=position_ids, attention_mask=attention_mask)
    return outputs.logits

# Custom forward function
def custom_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs, token_type_ids=token_type_ids,
                   position_ids=position_ids, attention_mask=attention_mask)
    return pred[:, position]

# Construct input and reference pairs
def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)

# Summarize attributions
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

# Compute attributions using Integrated Gradients
def compute_attributions(input_text, target_index):
    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id

    input_ids, ref_input_ids = construct_input_ref_pair(input_text, ref_token_id, sep_token_id, cls_token_id)
    attention_mask = torch.ones_like(input_ids, device=device)
    token_type_ids = torch.zeros_like(input_ids, device=device)

    lig = LayerIntegratedGradients(custom_forward_func, model.bert.embeddings)
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(token_type_ids, None, attention_mask, target_index),
                                        return_convergence_delta=True)
    summarized_attributions = summarize_attributions(attributions)
    return summarized_attributions, delta

# Visualize the attributions
def visualize_attributions(input_text, attributions, output_path):
    tokens = tokenizer.tokenize(input_text)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]  # Add special tokens
    scores = attributions.cpu().detach().numpy()

    plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
    colors = ['red' if score < 0 else 'green' for score in scores]
    plt.bar(range(len(scores)), scores, color=colors)
    plt.xticks(range(len(scores)), tokens, rotation=90)
    plt.xlabel('Tokens')
    plt.ylabel('Attribution Scores')
    plt.title('Word Importance Attribution')
    plt.tight_layout()  # Adjust subplot parameters to fit the plot within the figure area
    plt.subplots_adjust(bottom=0.3)  # Add extra space below the x-axis
    plt.savefig(output_path)
    plt.close()

with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

utterances = []
for k in transcriptions.keys():
    for u in transcriptions[k]['low_acc']:
        utterances.append(u)

# Dictionaries to store token attributions for CREWMATE and IMPOSTOR
token_attributions_crewmate = defaultdict(list)
token_attributions_impostor = defaultdict(list)

# Use all utterances to calculate token attributions
for utterance in tqdm(utterances, desc="Calculating attributions for all utterances"):
    inputs = tokenizer(utterance, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = predict(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        attributions, delta = compute_attributions(utterance, 0)  # CREWMATE
        tokens = tokenizer.tokenize(utterance)
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        scores = attributions.cpu().detach().numpy()
        for token, score in zip(tokens, scores):
            token_attributions_crewmate[token].append(score)
    elif predicted_class == 1:
        attributions, delta = compute_attributions(utterance, 1)  # IMPOSTOR
        tokens = tokenizer.tokenize(utterance)
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        scores = attributions.cpu().detach().numpy()
        for token, score in zip(tokens, scores):
            token_attributions_impostor[token].append(score)

# Calculate average attribution for each token for both classes
average_token_attributions_crewmate = {token: np.mean(scores) for token, scores in token_attributions_crewmate.items()}
average_token_attributions_impostor = {token: np.mean(scores) for token, scores in token_attributions_impostor.items()}

# Save the dictionaries with average contributions
os.makedirs('data/feature_importance', exist_ok=True)
with open('data/feature_importance/average_token_attributions_crewmate.pkl', 'wb') as f:
    pkl.dump(average_token_attributions_crewmate, f)

with open('data/feature_importance/average_token_attributions_impostor.pkl', 'wb') as f:
    pkl.dump(average_token_attributions_impostor, f)

# Sort tokens by their average attribution for both classes
sorted_tokens_crewmate = sorted(average_token_attributions_crewmate.items(), key=lambda item: item[1], reverse=True)
sorted_tokens_impostor = sorted(average_token_attributions_impostor.items(), key=lambda item: item[1], reverse=True)

# Display and save top tokens for CREWMATE with their average attributions
top_tokens_crewmate = sorted_tokens_crewmate[:20]
bottom_tokens_crewmate = sorted_tokens_crewmate[-20:]

print("Top tokens by average attribution for CREWMATE:")
for token, avg_attr in top_tokens_crewmate:
    print(f"Token: {token}, Average Attribution: {avg_attr}")

# Plot top tokens for CREWMATE
top_tokens_crewmate_names, top_tokens_crewmate_scores = zip(*top_tokens_crewmate)
plt.figure(figsize=(12, 6))
plt.bar(top_tokens_crewmate_names, top_tokens_crewmate_scores, color='green')
plt.xlabel('Tokens')
plt.ylabel('Average Attribution')
plt.title('Top 20 Tokens by Average Attribution for CREWMATE')
plt.xticks(rotation=90)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space below the x-axis
plt.savefig('data/feature_importance/top_tokens_crewmate_by_average_attribution.png')
plt.close()

print("\nBottom tokens by average attribution for CREWMATE:")
for token, avg_attr in bottom_tokens_crewmate:
    print(f"Token: {token}, Average Attribution: {avg_attr}")

# Plot bottom tokens for CREWMATE
bottom_tokens_crewmate_names, bottom_tokens_crewmate_scores = zip(*bottom_tokens_crewmate)
plt.figure(figsize=(12, 6))
plt.bar(bottom_tokens_crewmate_names, bottom_tokens_crewmate_scores, color='red')
plt.xlabel('Tokens')
plt.ylabel('Average Attribution')
plt.title('Bottom 20 Tokens by Average Attribution for CREWMATE')
plt.xticks(rotation=90)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space below the x-axis
plt.savefig('data/feature_importance/bottom_tokens_crewmate_by_average_attribution.png')
plt.close()

# Display and save top tokens for IMPOSTOR with their average attributions
top_tokens_impostor = sorted_tokens_impostor[:20]
bottom_tokens_impostor = sorted_tokens_impostor[-20:]

print("\nTop tokens by average attribution for IMPOSTOR:")
for token, avg_attr in top_tokens_impostor:
    print(f"Token: {token}, Average Attribution: {avg_attr}")

# Plot top tokens for IMPOSTOR
top_tokens_impostor_names, top_tokens_impostor_scores = zip(*top_tokens_impostor)
plt.figure(figsize=(12, 6))
plt.bar(top_tokens_impostor_names, top_tokens_impostor_scores, color='green')
plt.xlabel('Tokens')
plt.ylabel('Average Attribution')
plt.title('Top 20 Tokens by Average Attribution for IMPOSTOR')
plt.xticks(rotation=90)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space below the x-axis
plt.savefig('data/feature_importance/top_tokens_impostor_by_average_attribution.png')
plt.close()

print("\nBottom tokens by average attribution for IMPOSTOR:")
for token, avg_attr in bottom_tokens_impostor:
    print(f"Token: {token}, Average Attribution: {avg_attr}")

# Plot bottom tokens for IMPOSTOR
bottom_tokens_impostor_names, bottom_tokens_impostor_scores = zip(*bottom_tokens_impostor)
plt.figure(figsize=(12, 6))
plt.bar(bottom_tokens_impostor_names, bottom_tokens_impostor_scores, color='red')
plt.xlabel('Tokens')
plt.ylabel('Average Attribution')
plt.title('Bottom 20 Tokens by Average Attribution for IMPOSTOR')
plt.xticks(rotation=90)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Add extra space below the x-axis
plt.savefig('data/feature_importance/bottom_tokens_impostor_by_average_attribution.png')
plt.close()

# Select 50 random utterances for CREWMATE and IMPOSTOR to visualize feature importance
random.shuffle(utterances)
crewmate_utterances = []
impostor_utterances = []

# Select utterances until we have 50 for each class
for utterance in tqdm(utterances, desc="Selecting utterances for visualization"):
    if len(crewmate_utterances) >= 50 and len(impostor_utterances) >= 50:
        break

    inputs = tokenizer(utterance, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = predict(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0 and len(crewmate_utterances) < 50:
        crewmate_utterances.append(utterance)
    elif predicted_class == 1 and len(impostor_utterances) < 50:
        impostor_utterances.append(utterance)

# Generate and visualize attributions for selected utterances
for i, utterance in enumerate(tqdm(crewmate_utterances, desc="Visualizing CREWMATE attributions")):
    attributions, delta = compute_attributions(utterance, 0)  # CREWMATE
    visualize_attributions(utterance, attributions, f'data/feature_importance/crewmate_attribution_{i}.png')

for i, utterance in enumerate(tqdm(impostor_utterances, desc="Visualizing IMPOSTOR attributions")):
    attributions, delta = compute_attributions(utterance, 1)  # IMPOSTOR
    visualize_attributions(utterance, attributions, f'data/feature_importance/impostor_attribution_{i}.png')


# analyze token of question mark
with open('data/feature_importance/average_token_attributions_crewmate.pkl', 'rb') as file:
    crewmate_attr = pkl.load(file)

with open('data/feature_importance/average_token_attributions_impostor.pkl', 'rb') as file:
    impostor_attr = pkl.load(file)

print(crewmate_attr['?'])
