import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import kaldiio
import argparse as ap

from sklearn.metrics import average_precision_score

import numpy as np
import os
from collections import Counter  # Import Counter

from personal_vad import PersonalVAD, WPL, pad_collate

# model hyper parameters
num_epochs = 10
batch_size = 64
batch_size_test = 64

input_dim = 297
hidden_dim = 64
out_dim = 3
num_layers = 2
lr = 1e-3
SCHEDULER = True

DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
EMBED_PATH = 'embeddings'
MODEL_PATH = 'vad_set.pt'
SAVE_MODEL = True

USE_WPL = True
NUM_WORKERS = 2

# Selects which of the scoring methods should be used...
# legend: scores[0,:] -> baseline, 1 -> partially-constant, 2 -> linearly-interpolated
SCORE_TYPE = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WPL_WEIGHTS = torch.tensor([1.0, 0.1, 1.0]).to(device)

class VadSETDataset(Dataset):
    """VadSET dataset class. Uses kaldi scp and ark files."""

    def __init__(self, root_dir, embed_path, score_type):
        self.root_dir = root_dir
        self.embed_path = embed_path
        self.score_type = score_type

        # load the scp files...
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        self.scores = kaldiio.load_scp(f'{self.root_dir}/scores.scp')
        self.labels = kaldiio.load_scp(f'{self.root_dir}/labels.scp')
        self.keys = np.array(list(self.fbanks))  # get all the keys
        self.embed = kaldiio.load_scp(f'{self.embed_path}/dvectors.scp')

        # load the target speaker ids
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

        # add the speaker verification scores array to the feature vector
        x = np.hstack((x, np.expand_dims(scores, 1)))

        # add the dvector array to the feature vector
        x = np.hstack((x, np.full((x.shape[0], 256), embed)))

        # Create writable copies of the arrays before converting to tensors
        x = torch.from_numpy(np.copy(x)).float()
        y = torch.from_numpy(np.copy(y)).long()
        return x, y

def count_labels(dataset):
    """Function to count label occurrences in the dataset."""
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels.tolist())
    return Counter(all_labels)

if __name__ == '__main__':
    """ Model training  """

    # program arguments
    parser = ap.ArgumentParser(description="Train the VAD SET model.")
    parser.add_argument('--train_dir', type=str, default=DATA_TRAIN)
    parser.add_argument('--test_dir', type=str, default=DATA_TEST)
    parser.add_argument('--embed_path', type=str, default=EMBED_PATH)
    parser.add_argument('--score_type', type=int, default=SCORE_TYPE)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--linear', type=bool, default=False)
    parser.add_argument('--use_fc', type=bool, default=False)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--use_kaldi', action='store_true')
    parser.add_argument('--use_wpl', action='store_true')
    parser.add_argument('--nsave_model', action='store_false')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATA_TRAIN = args.train_dir
    DATA_TEST = args.test_dir
    EMBED_PATH = args.embed_path
    SCORE_TYPE = args.score_type
    linear = args.linear
    USE_WPL = args.use_wpl
    SAVE_MODEL = args.nsave_model
    MODEL = args.model
    use_fc = args.use_fc

    if SCORE_TYPE not in [0, 1, 2]:
        print(f"Error: invalid scoring type: {SCORE_TYPE}. The values have to be in {0, 1, 2}.")
        sys.exit(1)

    # Load the data and create DataLoader instances
    train_data = VadSETDataset(DATA_TRAIN, EMBED_PATH, SCORE_TYPE)
    test_data = VadSETDataset(DATA_TEST, EMBED_PATH, SCORE_TYPE)


    # Count the occurrences of each label in the dataset
    train_labels_count = count_labels(train_data)
    test_labels_count = count_labels(test_data)

    print("Training labels distribution:", train_labels_count)
    print("Testing labels distribution:", test_labels_count)

    train_loader = DataLoader(
            dataset=train_data, num_workers=NUM_WORKERS, pin_memory=True,
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(
            dataset=test_data, num_workers=NUM_WORKERS, pin_memory=True,
            batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=use_fc, linear=linear).to(device)

    # Load the pretrained model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded pretrained model from {MODEL_PATH}")

    USE_WPL = True
    if USE_WPL:
        criterion = WPL(WPL_WEIGHTS)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    softmax = nn.Softmax(dim=1)

    # Train!!! hype!!!
    for epoch in range(num_epochs):
        print(f"====== Starting epoch {epoch} ======")
        for batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):
            y_padded = y_padded.to(device)

            # pass the data through the model
            out_padded, _ = model(x_padded.to(device), x_lens, None)

            # compute the loss
            loss = 0
            for j in range(out_padded.size(0)):
                loss += criterion(out_padded[j][:y_lens[j]], y_padded[j][:y_lens[j]])

            loss /= batch_size # normalize for the batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 10 == 0:
                print(f'Batch: {batch}, loss = {loss:.4f}')

        if SCHEDULER and epoch < 2:
            scheduler.step() # learning rate adjust
            if epoch == 1:
                optimizer.param_groups[0]['lr'] = 5e-5
        if SCHEDULER and epoch == 7:
            optimizer.param_groups[0]['lr'] = 1e-5

        # Test the model after each epoch
        with torch.no_grad():
            print("testing...")
            n_correct = 0
            n_samples = 0
            targets = []
            outputs = []
            predictions = []
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = model(x_padded.to(device), x_lens, None)


                # value, index
                for j in range(out_padded.size(0)):
                    classes = torch.argmax(out_padded[j][:y_lens[j]], dim=1)
                    predictions.extend(classes.cpu().numpy())
                    n_samples += y_lens[j]
                    n_correct += torch.sum(classes == y_padded[j][:y_lens[j]]).item()

                    # average precision
                    p = softmax(out_padded[j][:y_lens[j]])
                    outputs.append(p.cpu().numpy())
                    targets.append(y_padded[j][:y_lens[j]].cpu().numpy())

            acc = 100.0 * n_correct / n_samples
            print(f"accuracy = {acc:.2f}")

            # Calculate prediction statistics
            prediction_counts = Counter(predictions)
            for cls, count in prediction_counts.items():
                print(f"Class {cls} was predicted {count} times")

            # and run the AP
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)
            targets_oh = np.eye(3)[targets]
            out_AP = average_precision_score(targets_oh, outputs, average=None)
            mAP = average_precision_score(targets_oh, outputs, average='micro')

            print(out_AP)
            print(f"mAP: {mAP}")

        # Save the model - after each epoch for ensurance...
        if SAVE_MODEL:

            if SAVE_MODEL:
                torch.save(model.state_dict(), f'data/Fine-TunedModels/{MODEL_PATH}')

