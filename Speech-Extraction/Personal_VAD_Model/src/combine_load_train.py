import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import kaldiio
import argparse as ap
from collections import Counter
import numpy as np
import os
import sys
import pickle
import time
import logging

from personal_vad import pad_collate

log_file = 'training_log.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

start_time = time.time()

# Model hyperparameters
num_epochs = 10
batch_size = 1
input_dim = 297
hidden_dim = 64
out_dim = 3  # Three labels: 0, 1, 2
num_layers = 2
lr = 1e-3
SCHEDULER = True

DATA_TRAIN = 'data/features_demo'
EMBED_PATH = 'data/embeddings'
SAVE_MODEL = True
USE_WPL = False
NUM_WORKERS = 2
SCORE_TYPE = 0

# Set the visible CUDA devices to the second GPU (index 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# os.chdir('data/eval_dir')
logging.info(os.getcwd())

class VadSETDataset(Dataset):
    """VadSET dataset class. Uses kaldi scp and ark files."""

    def __init__(self, root_dir, embed_path, score_type):
        self.root_dir = root_dir
        self.embed_path = embed_path
        self.score_type = score_type

        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        self.scores = kaldiio.load_scp(f'{self.root_dir}/scores.scp')
        self.labels = kaldiio.load_scp(f'{self.root_dir}/labels.scp')
        self.keys = np.array(list(self.fbanks))
        self.embed = kaldiio.load_scp(f'{self.embed_path}/dvectors.scp')

        self.targets = {}
        with open(f'{self.root_dir}/targets.scp') as targets:
            for line in targets:
                (utt_id, target) = line.split()
                self.targets[utt_id] = target

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        target = self.targets[key]
        x = self.fbanks[key]
        scores = self.scores[key][self.score_type, :]
        embed = self.embed[target]
        y = self.labels[key]

        x = np.hstack((x, np.expand_dims(scores, 1)))
        x = np.hstack((x, np.full((x.shape[0], 256), embed)))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        return x, y

# Model Architecture
class PersonalVAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, use_fc=True, linear=False):
        super(PersonalVAD, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.use_fc = use_fc
        self.linear = linear

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        if use_fc:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            if not self.linear:
                self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, x_lens, hidden=None):
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

        if self.use_fc:
            out_padded = self.fc1(out_padded)
            if not self.linear:
                out_padded = self.tanh(out_padded)

        out_padded = self.fc2(out_padded)
        return out_padded, hidden


# Instantiate the model
model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim).to(device)

# load model already trained for six epochs
model.load_state_dict(torch.load('data/vad_set_ut100000.pt', map_location=device))
model.eval()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.param_groups[0]['lr'] = 5e-5  # because i start at epoch=6

# Load the data and create DataLoader instances
train_data = VadSETDataset(DATA_TRAIN, EMBED_PATH, SCORE_TYPE)
train_loader = DataLoader(
    dataset=train_data, num_workers=NUM_WORKERS, pin_memory=True,
    batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training Loop
for epoch in range(6, num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = {0: 0, 1: 0, 2: 0}
    total_counts = {0: 0, 1: 0, 2: 0}
    pred_counts = {0: 0, 1: 0, 2: 0}

    true_positive = {0: 0, 1: 0, 2: 0}
    true_negative = {0: 0, 1: 0, 2: 0}
    false_positive = {0: 0, 1: 0, 2: 0}
    false_negative = {0: 0, 1: 0, 2: 0}

    logging.info(f"====== Starting epoch {epoch} ======")
    for batch, (inputs, labels, input_lengths, _) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs, _ = model(inputs, input_lengths)
        outputs = outputs.view(-1, out_dim)
        labels = labels.view(-1)
        #print(f'batch: {batch}')
        #print(f'labels: {labels}')
        label_counts = torch.bincount(labels, minlength=3)  # Ensure we count for 0, 1, and 2

        # Print the counts for each label
        #print(f'Number of 0s: {label_counts[0].item()}')
        #print(f'Number of 1s: {label_counts[1].item()}')
        #print(f'Number of 2s: {label_counts[2].item()}')

        # Get predicted labels
        _, predicted_labels = torch.max(outputs, 1)

        # Update counts
        for label in [0, 1, 2]:
            total_counts[label] += (labels == label).sum().item()
            total_correct[label] += ((predicted_labels == label) & (labels == label)).sum().item()
            pred_counts[label] += (predicted_labels == label).sum().item()

            true_positive[label] += ((predicted_labels == label) & (labels == label)).sum().item()
            true_negative[label] += ((predicted_labels != label) & (labels != label)).sum().item()
            false_positive[label] += ((predicted_labels == label) & (labels != label)).sum().item()
            false_negative[label] += ((predicted_labels != label) & (labels == label)).sum().item()

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()  # Accumulate the loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print results for each epoch
    for label in [0, 1, 2]:
        logging.info(f'Label {label}: Predicted {pred_counts[label]} times')
        logging.info(f'Label {label} occurred {total_counts[label]} times')

        accuracy = (true_positive[label] + true_negative[label]) / (true_positive[label] + true_negative[label] + false_positive[label] + false_negative[label])
        precision = true_positive[label] / (true_positive[label] + false_positive[label])
        recall = true_positive[label] / (true_positive[label] + false_negative[label])
        logging.info(f'Label {label}: Accuracy: {accuracy}')
        logging.info(f'Label {label}: Precision: {precision}')
        logging.info(f'Label {label}: Recall: {recall}')
    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}\n')
    logging.info('--------')

    if SCHEDULER and epoch < 2:
        scheduler.step()  # learning rate adjust
        if epoch == 1:
            optimizer.param_groups[0]['lr'] = 5e-5
    if SCHEDULER and epoch == 7:
        optimizer.param_groups[0]['lr'] = 1e-5

    # Save the model after each epoch for certainty
    torch.save(model.state_dict(), f'data/vad_set_ut100000_{epoch + 1}.pt')

logging.info('Training complete.')

# save final model
torch.save(model.state_dict(), f'data/vad_set_ut100000_final.pt')

end_time = time.time()
duration = end_time - start_time
logging.info(f"Script executed in {duration:.2f} seconds")
