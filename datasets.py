import os
import time
import torch
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from torch.utils import data
from torch_stft.stft_implementation.stft import STFT

BLANK_TOKEN_ID = 1024
TIME_STEP = 2

class MyDataset(data.Dataset):
    def __init__(self, tokenizer, train_manifest, file_num, min_len, max_len, n_fft=512, hop_length=256, win_length=512):
        super().__init__()

        self.final_df = self.get_files_from_manifest(train_manifest)
        self.root_audio_folder = Path(train_manifest).parent.absolute().as_posix()
        self.final_df = self.final_df[(self.final_df['duration'] > min_len) & (self.final_df['duration'] < max_len)]
        if file_num > 0:
            self.final_df = self.final_df.head(file_num)
        print(f"Dataset contain {self.final_df["duration"].sum() / 3600:.2f} hours")
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sampling_rate = 16000

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

    @staticmethod
    def get_files_from_manifest(manifest):
        df = pd.read_json(manifest, lines=True)
        df = df[["audio_filepath", "text", "duration"]]

        return df
    
    def explicit_shuffle(self):
        print("Shuffle dataset")
        self.final_df = self.final_df.sample(frac=1, random_state=time.time_ns() % 2**32).reset_index(drop=True)

    def __getitem__(self, idx):
        file = self.final_df.iloc[idx]
        audio_file_path = os.path.join(self.root_audio_folder, file["audio_filepath"])
        audio, sr = librosa.load(audio_file_path)
        if sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
        encoded_ids = self.tokenizer.encode(file["text"], out_type=int)
        encoded_ids = [el for el in encoded_ids if el != 0] # Drop unknown tokens for target, because models must not predict it

        audio = self.stft.transform(torch.from_numpy(audio).float().unsqueeze(0))[0]
        audio = audio.transpose(0, 2)
        time_patch = TIME_STEP
        need_pad = (audio.shape[-2] + time_patch) // time_patch * time_patch - audio.shape[-2]
        audio = torch.cat((audio, torch.zeros(audio.shape[0], need_pad, audio.shape[-1])), dim=1)

        return audio, torch.tensor(encoded_ids)

    def __len__(self):
        return len(self.final_df)


def custom_collate_fn(batch):
    # 'batch' is a list of samples from MyDataset.__getitem__
    # In this example, each sample is a tensor of variable length

    # Get lengths of sequences
    audio_len = []
    ids_len = []
    for audio, ids in batch:
        audio_len.append(audio.shape[-2])
        ids_len.append(ids.shape[-1])

    max_audio_len = max(audio_len)
    max_ids_len = max(ids_len)

    # Pad sequences to the max_len
    audio_padding_len = []
    padded_audio = []

    ids_len = []
    padded_ids = []
    for audio, ids in batch:
        time_padding = max_audio_len - audio.shape[-2]
        padded_item = torch.cat((audio, torch.zeros(audio.shape[0], time_padding, audio.shape[-1],)), dim=1)
        audio_padding_len.append(audio.shape[-2])
        padded_audio.append(padded_item)

        ids_padding = int(max_ids_len - ids.shape[-1])
        padded_item = torch.cat((ids, torch.full(size=(ids_padding,), fill_value=BLANK_TOKEN_ID)), dim=0)
        ids_len.append(ids.shape[-1])
        padded_ids.append(padded_item)

    # Stack padded sequences and return lengths
    return torch.stack(padded_audio), torch.tensor(audio_padding_len), torch.stack(padded_ids), torch.tensor(ids_len)
