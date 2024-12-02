import kaldiio
import os

# Path to the scp file
data_dir = '/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model'

os.chdir(data_dir)

# Path to the scp file (now relative to the data directory)
scp_path = 'data/embeddings/dvectors.scp'

# Read the scp file
dvector_dict = kaldiio.load_scp(scp_path)

# dvector_dict is a dictionary where the keys are the utterance IDs and the values are the embeddings
l = []
for utt_id, dvector in dvector_dict.items():
    print(f'Utt ID: {utt_id}')
    print(f'D-vector; {dvector}')
    print(f'D-vector shape: {dvector.shape}')


# Path to the scp file (now relative to the data directory)
scp_path = 'data/features_demo/fbanks.scp'

# Read the scp file
file = kaldiio.load_scp(scp_path)

for utt_id, dvector in file.items():
    print(utt_id)
    print(dvector.shape)

with open(scp_path, 'r') as file:
    for line in file:
        print(line.strip())

os.getcwd()

"""
scp_path = 'data/features_demo/labels.scp'

# Read the scp file
dict = kaldiio.load_scp(scp_path)

for i in dict['he_jv_sk_ka_tc']:
    print(i)

targets = kaldiio.load_scp('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model/data/features_demo/targets.scp')
"""
