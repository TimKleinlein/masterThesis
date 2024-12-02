import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle as pkl
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


# first bring dictionary in format: {'disc_round_uttNr': {'text': xxx, 'label': X}, ...}
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

"""
data = {}
for dr in transcriptions.keys():
    for ind, utt in enumerate(transcriptions[dr]['high_acc']):
        data[f'{dr}_{ind}'] = {'text': utt, 'label': transcriptions[dr]['role']}
"""
data = {}
for dr in transcriptions.keys():
    if transcriptions[dr]['high_acc']:
        data[dr] = {'text': '. '.join(transcriptions[dr]['high_acc']), 'label': transcriptions[dr]['role']}


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

# remove neutral roles from data
new_data = {}
for k in data.keys():
    if data[k]['label'] != 'NEUTRAL':
        new_data[k] = data[k]

data = new_data

iNr = 0
cNr = 0
for k in data.keys():
    if data[k]['label'] == 'IMPOSTOR':
        iNr += 1
    elif data[k]['label'] == 'CREWMATE':
        cNr += 1

# Tokenizer
tokenizer = get_tokenizer('basic_english')


# Build the vocabulary
def yield_tokens(data_iter):
    for item in data_iter.values():
        yield tokenizer(item['text'])


vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])


# Custom dataset class
class CustomTextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')
        self.labels = {label: idx for idx, label in enumerate(set(item['label'] for item in data.values()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = list(self.data.values())[idx]
        tokens = self.tokenizer(item['text'])
        numerical_tokens = [self.vocab[token] for token in tokens]
        label = self.labels[item['label']]
        return torch.tensor(numerical_tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# Print label mappings
labels = {label: idx for idx, label in enumerate(set(item['label'] for item in data.values()))}
print("Label mappings:")
for label, idx in labels.items():
    print(f"Label: {label}, Index: {idx}")

# Custom collate function
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    labels = torch.stack(labels)
    return texts_padded, labels


# Define the model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


# Training parameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
num_classes = len(set(item['label'] for item in data.values()))  # number of unique labels
batch_size = 1
num_epochs = 10
learning_rate = 0.001

# Create the model
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
# Calculate class weights
counter = Counter(item['label'] for item in data.values())
total_count = sum(counter.values())
class_weights = {cls: total_count / count for cls, count in counter.items()}
weights = torch.tensor([class_weights[cls] for cls in labels.keys()], dtype=torch.float).to(device)

# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create data loaders for the training and validation sets
train_size = int(0.8 * len(data))
valid_size = len(data) - train_size
train_data, valid_data = random_split(list(data.items()), [train_size, valid_size])

train_dataset = CustomTextDataset(dict(train_data), vocab)
valid_dataset = CustomTextDataset(dict(valid_data), vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    label_counts = {idx: 0 for idx in labels.values()}
    for inputs, targets in train_loader:

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # Count predictions
        _, predicted = torch.max(outputs, 1)
        for label in predicted.cpu().numpy():
            label_counts[label] += 1

    # Evaluate on the validation set after every epoch
    model.eval()
    total_val_loss = 0.0
    total_val_samples = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)

            total_val_loss += val_loss.item() * inputs.size(0)
            total_val_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_val_loss = total_val_loss / total_val_samples

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"Label counts in epoch {epoch + 1}: {label_counts}")


# Evaluation
def evaluate_model(model, data_loader):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    return precision, recall, f1


precision, recall, f1 = evaluate_model(model, valid_loader)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
