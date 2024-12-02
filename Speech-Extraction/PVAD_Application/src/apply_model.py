import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import kaldiio
import numpy as np
import os
import logging
import pickle

# Ensure you import or define the pad_collate_inference function
# The modified function specific to inference
def pad_collate_inference(batch):
    (xx, keys, lengths) = zip(*batch)
    x_lens = [len(x) for x in xx]
    x_padded = pad_sequence(xx, batch_first=True, padding_value=0)
    return x_padded, keys, x_lens, lengths


# Model hyperparameters
input_dim = 297
hidden_dim = 64
out_dim = 3  # Three labels: 0, 1, 2
num_layers = 2

DATA_APPLICATION = 'data/features_application'
EMBED_PATH = 'data/embeddings'
MODEL_PATH = 'data/vad_set_ut100000.pt'
NUM_WORKERS = 2
SCORE_TYPE = 0
OUTPUT_DIR = 'data/pvad_output_files'  # Directory to save the output pickle files
FRAME_LENGTH_SECONDS = 0.01  # 10ms per frame
PROB_THRESHOLD = 0.95  # Probability threshold for label 2

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set the visible CUDA devices to the second GPU (index 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class VadSETDataset(Dataset):
    """VadSET dataset class. Uses kaldi scp and ark files."""

    def __init__(self, root_dir, embed_path, score_type):
        self.root_dir = root_dir
        self.embed_path = embed_path
        self.score_type = score_type

        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        self.scores = kaldiio.load_scp(f'{self.root_dir}/scores.scp')
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
        x = np.hstack((x, np.expand_dims(scores, 1)))
        x = np.hstack((x, np.full((x.shape[0], 256), embed)))

        x = torch.from_numpy(x).float()
        return x, key, x.shape[0]  # Also return the length of the feature sequence


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


# Load the pre-trained model
model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load the data and create DataLoader instances
application_data = VadSETDataset(DATA_APPLICATION, EMBED_PATH, SCORE_TYPE)
application_loader = DataLoader(
    dataset=application_data, num_workers=NUM_WORKERS, pin_memory=True,
    batch_size=1, shuffle=False, collate_fn=pad_collate_inference)

# Inference Loop with Logging All Probabilities
with torch.no_grad():
    for batch, (inputs, keys, input_lengths, lengths) in enumerate(application_loader):
        inputs = inputs.to(device)

        # Forward pass
        outputs, _ = model(inputs, input_lengths)
        outputs = outputs.view(-1, out_dim)

        # Apply softmax to get probabilities
        probabilities = nn.functional.softmax(outputs, dim=1)

        for i, key in enumerate(keys):
            file_lengths = FRAME_LENGTH_SECONDS * lengths[i]
            frame_probabilities = probabilities.view(inputs.size(1), out_dim).cpu().numpy()
            frame_intervals = [(FRAME_LENGTH_SECONDS * j, FRAME_LENGTH_SECONDS * (j + 1)) for j in
                               range(frame_probabilities.shape[0])]

            predictions = {interval: probs.tolist() for interval, probs in zip(frame_intervals, frame_probabilities)}

            # Collect timestamps where the label is not 2 or the probability of label 2 is less than the threshold
            label_not_2_or_low_prob_2_timestamps = [
                [start, end] for (start, end), probs in zip(frame_intervals, frame_probabilities)
                if np.argmax(probs) != 2 or probs[2] < PROB_THRESHOLD
            ]

            # Save predictions dictionary to a pickle file
            predictions_file = os.path.join(OUTPUT_DIR, f'{key}_labeled_intervals.pkl')
            with open(predictions_file, 'wb') as f:
                pickle.dump(predictions, f)

            # Save label dictionary to a pickle file
            label_dict_file = os.path.join(OUTPUT_DIR, f'{key}_nontarget_label_dict.pkl')
            with open(label_dict_file, 'wb') as f:
                pickle.dump(label_not_2_or_low_prob_2_timestamps, f)
