import os
import numpy as np
import kaldiio
import librosa
import argparse as ap
import torch
from kaldiio import ReadHelper, WriteHelper
from glob import glob
import multiprocessing as mp
import pickle

from resemblyzer import VoiceEncoder
from resemblyzer_mod import VoiceEncoderMod


# Define constants for paths
KALDI_ROOT = 'kaldi/egs/pvad/'
DATA_ROOT = 'data/'
EMBED_PATH = 'data/embeddings/'
DEST = 'data/features_application/'
EMBED = 'data/embeddings/'

# Load abbreviation dictionaries
with open('data/abbr_dict_reverse.pkl', 'rb') as pkl_file:
    abbr_dict_reverse = pickle.load(pkl_file)

with open('data/abbr_dict.pkl', 'rb') as pkl_file:
    abbr_dict = pickle.load(pkl_file)

# Resemblyzer d-vector extraction parameters
rate = 2.5
samples_per_frame = 160
frame_step = int(np.round((16000 / rate) / samples_per_frame))
min_coverage = 0.5

def cos(a, b):
    """Compute the cosine similarity of two vectors"""
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

def load_dvector(spk_id, embed_scp):
    """Load the dvector for the target speaker."""
    embedding = embed_scp[spk_id]
    return embedding, spk_id

def gpu_worker(q_send, q_return):
    """GPU worker process."""
    encoder = VoiceEncoderMod()
    device = encoder.device

    while True:
        fbanks, fbanks_sliced, pid = q_send.get()
        fbanks = fbanks.to(device)
        fbanks_sliced = fbanks_sliced.to(device)

        with torch.no_grad():
            embeds_stream, _ = encoder.forward_stream(fbanks, None)
            embeds_stream = embeds_stream.cpu()
            embeds_slices = encoder(fbanks_sliced).cpu()

        q_return[pid].put((embeds_stream, embeds_slices))

def extract_features(scp, q_send, q_return):
    """CPU worker PVAD feature extraction process."""
    wav_scp = ReadHelper(f'scp:{scp}')
    pid = int(scp.rpartition('.')[0].rpartition('_')[2])
    array_writer = WriteHelper(f'ark,scp:{DEST}/fbanks_{pid}.ark,{DEST}/fbanks_{pid}.scp')
    score_writer = WriteHelper(f'ark,scp:{DEST}/scores_{pid}.ark,{DEST}/scores_{pid}.scp')
    target_writer = open(f'{DEST}/targets_{pid}.scp', 'w')
    embed_scp = kaldiio.load_scp(f'{EMBED}/dvectors.scp')

    for ind, (utt_id, (sr, arr)) in enumerate(wav_scp):
        print(utt_id)
        arr = arr.astype(np.float32, order='C') / 32768

        fbanks = librosa.feature.melspectrogram(y=arr, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype('float32').T[:-2]
        logfbanks = np.log10(fbanks + 1e-6)

        wav = arr.copy()
        wav_slices, mel_slices = VoiceEncoder.compute_partial_slices(wav.size, rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= wav.size:
            wav = np.pad(arr, (0, max_wave_length - wav.size), "constant")
        mels = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype('float32').T

        fbanks_sliced = np.array([mels[s] for s in mel_slices])

        # Extract speaker ID from utterance name
        spk_id = utt_id.split('_')[-1]
        spk_embed, spk_id = load_dvector(spk_id, embed_scp)

        fbanks_tensor = torch.unsqueeze(torch.from_numpy(fbanks), 0)
        q_send.put((fbanks_tensor, torch.from_numpy(fbanks_sliced), pid))
        embeds_stream, embeds_slices = q_return.get()

        embeds_stream = embeds_stream.numpy().squeeze()
        embeds_slices = embeds_slices.numpy()

        # Compute cosine similarities for scores
        scores_slices = np.array([cos(spk_embed, cur_embed) for cur_embed in embeds_slices])
        scores_stream = np.array([cos(spk_embed, cur_embed) for cur_embed in embeds_stream])

        scores_kron = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
        if scores_slices.size > 1:
            scores_kron = np.append(scores_kron, np.kron(scores_slices[1:], np.ones(frame_step, dtype='float32')))
        scores_kron = scores_kron[:logfbanks.shape[0]]

        scores_lin = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
        for i, s in enumerate(scores_slices[1:]):
            scores_lin = np.append(scores_lin, np.linspace(scores_slices[i], s, frame_step, endpoint=False))
        scores_lin = scores_lin[:logfbanks.shape[0]]

        scores = np.stack((scores_stream, scores_kron, scores_lin))



        array_writer(utt_id, logfbanks)
        score_writer(utt_id, scores)
        target_writer.write(f"{utt_id} {spk_id}\n")

        if ind % 100 == 0:
            array_writer.fark.flush()
            array_writer.fscp.flush()
            score_writer.fark.flush()
            score_writer.fscp.flush()
            target_writer.flush()

    wav_scp.close()
    array_writer.close()
    score_writer.close()
    target_writer.close()

if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Extract speaker embeddings for the LibriSpeech dataset.")
    parser.add_argument('--kaldi_root', type=str, required=False, default=KALDI_ROOT, help="Specify the Kaldi pvad project root path")
    parser.add_argument('--data_root', type=str, required=False, default=DATA_ROOT, help="Specify the directory, where the target .scp files are situated")
    parser.add_argument('--dest_path', type=str, required=False, default=DEST, help="Specify the feature output directory")
    parser.add_argument('--embed_path', type=str, required=False, default=EMBED, help="Specify the path to the embeddings folder")
    parser.add_argument('--use_kaldi', action='store_true', help="Set this flag if the source dataset was augmented with Kaldi.")
    args = parser.parse_args()

    USE_KALDI = args.use_kaldi
    KALDI_ROOT = args.kaldi_root
    DATA_ROOT = args.data_root
    DEST = args.dest_path
    EMBED = args.embed_path

    if USE_KALDI:
        os.chdir(KALDI_ROOT)

    files = glob(DATA_ROOT + '/split_*.scp')
    files.sort()
    nj = len(files)

    manager = mp.Manager()
    q_send = manager.Queue()
    q_return = [manager.Queue() for _ in range(nj)]

    worker = mp.Process(target=gpu_worker, args=(q_send, q_return,))
    worker.daemon = True
    worker.start()

    pool = mp.Pool(processes=nj)
    pool.starmap(extract_features, zip(files, [q_send] * nj, q_return))
    pool.close()
    pool.join()

