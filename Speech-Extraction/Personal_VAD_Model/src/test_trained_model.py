import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import kaldiio
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from personal_vad import PersonalVAD, pad_collate

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
        self.keys = np.array(list(self.fbanks)) # get all the keys
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
        scores = self.scores[key][self.score_type,:]
        embed = self.embed[target]
        y = self.labels[key]

        # add the speaker verification scores array to the feature vector
        x = np.hstack((x, np.expand_dims(scores, 1)))

        # add the dvector array to the feature vector
        x = np.hstack((x, np.full((x.shape[0], 256), embed)))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        return x, y

if __name__ == '__main__':

    # Model parameters
    input_dim = 297
    hidden_dim = 64
    num_layers = 2
    out_dim = 3
    use_fc = True  # Change this if needed
    linear = False  # Change this if needed

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the model
    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=use_fc, linear=linear).to(device)

    # Load the trained model
    model.load_state_dict(torch.load('data/vad_set_ut10000.pt'))
    model.eval()

    # Load new data
    new_data_dir = 'data/features_demo'
    embed_path = 'data/embeddings'
    score_type = 0  # Set the appropriate score type
    new_data = VadSETDataset(new_data_dir, embed_path, score_type)
    new_data_loader = DataLoader(dataset=new_data, batch_size=1, shuffle=False, collate_fn=pad_collate)

    # Variables to keep track of predictions and true labels
    all_predictions = []
    all_true_labels = []

    # Variables to keep track of label-specific counts
    label_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Make predictions
    with torch.no_grad():
        for x_padded, y_padded, x_lens, y_lens in new_data_loader:
            x_padded = x_padded.to(device)
            y_padded = y_padded.to(device)

            # Forward pass through the model
            out_padded, _ = model(x_padded, x_lens, None)

            # Collect predictions and true labels
            for i in range(out_padded.size(0)):  # Iterate over each sample in the batch
                for j in range(x_lens[i]):  # Iterate over each time step in the sample
                    predicted_class = torch.argmax(out_padded[i][j]).item()
                    print(f'pred: {predicted_class}')
                    true_label = y_padded[i][j].item()
                    print(f'true: {true_label}')
                    all_predictions.append(predicted_class)
                    all_true_labels.append(true_label)

                    # Update label-specific counts
                    label_counts[true_label]['total'] += 1
                    if predicted_class == true_label:
                        label_counts[true_label]['correct'] += 1

    # Calculate and print the accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.2f}")

    # Print the accuracy for each label
    for label in sorted(label_counts.keys()):
        correct = label_counts[label]['correct']
        total = label_counts[label]['total']
        label_accuracy = correct / total if total > 0 else 0
        print(f"Label {label}:")
        print(f"  Total predictions: {total}")
        print(f"  Correct predictions: {correct}")
        print(f"  Accuracy: {label_accuracy:.2f}")
