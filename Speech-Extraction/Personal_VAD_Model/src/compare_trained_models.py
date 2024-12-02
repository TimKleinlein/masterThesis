import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import kaldiio
import argparse as ap
from collections import Counter
import numpy as np
import os
import pickle
import time
import logging

from personal_vad import pad_collate

log_file = 'test_log.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

start_time = time.time()

# Model hyperparameters
batch_size = 1
input_dim = 297
hidden_dim = 64
out_dim = 3  # Three labels: 0, 1, 2
num_layers = 2
DATA_TRAIN = 'data/features_test'
EMBED_PATH = 'data/embeddings'
NUM_WORKERS = 2
SCORE_TYPE = 0

# Set the visible CUDA devices to the second GPU (index 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

# check for all trained models how they perform on test data
for trained_model in ['vad_set_ut100000_6.pt', 'vad_set_ut100000_7.pt', 'vad_set_ut100000_8.pt', 'vad_set_ut100000_9.pt', 'vad_set_ut100000_10.pt']:
    logging.info(f'Model {trained_model}')
    logging.info('--------')
    # Instantiate the model
    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim).to(device)
    # Load pre-trained model
    model.load_state_dict(torch.load(f'data/{trained_model}', map_location=device))
    model.eval()

    # Load the test data and create DataLoader instances
    test_data = VadSETDataset(DATA_TRAIN, EMBED_PATH, SCORE_TYPE)
    test_loader = DataLoader(
        dataset=test_data, num_workers=NUM_WORKERS, pin_memory=True,
        batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    total_loss = 0.0
    total_correct = {0: 0, 1: 0, 2: 0}
    total_counts = {0: 0, 1: 0, 2: 0}
    pred_counts = {0: 0, 1: 0, 2: 0}

    true_positive = {0: 0, 1: 0, 2: 0}
    true_negative = {0: 0, 1: 0, 2: 0}
    false_positive = {0: 0, 1: 0, 2: 0}
    false_negative = {0: 0, 1: 0, 2: 0}

    model.eval()
    with torch.no_grad():
        for batch, (inputs, labels, input_lengths, _) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(inputs, input_lengths)
            outputs = outputs.view(-1, out_dim)
            labels = labels.view(-1)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

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

    # Print results
    for label in [0, 1, 2]:
        logging.info(f'Label {label}: Predicted {pred_counts[label]} times')
        logging.info(f'Label {label} occurred {total_counts[label]} times')

        accuracy = (true_positive[label] + true_negative[label]) / (true_positive[label] + true_negative[label] + false_positive[label] + false_negative[label])
        precision = true_positive[label] / (true_positive[label] + false_positive[label])
        recall = true_positive[label] / (true_positive[label] + false_negative[label])
        logging.info(f'Label {label}: Accuracy: {accuracy}')
        logging.info(f'Label {label}: Precision: {precision}')
        logging.info(f'Label {label}: Recall: {recall}')

    logging.info(f'Loss: {total_loss / len(test_loader):.4f}')

end_time = time.time()
duration = end_time - start_time
logging.info(f"Script executed in {duration:.2f} seconds")
