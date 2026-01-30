import os
import time
import random
import math
import torch
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from torch.utils import data
from torch_stft.stft_implementation.stft import STFT

from audiomentations import Compose, PitchShift, HighPassFilter, LowPassFilter, AddGaussianSNR, ApplyImpulseResponse, \
    AddGaussianNoise, SevenBandParametricEQ, LoudnessNormalization, Gain, Mp3Compression

BLANK_TOKEN_ID = 1024


class MyDataset(data.Dataset):
    def __init__(self, tokenizer, train_manifest, file_num, min_len, max_len, n_fft=512, hop_length=256,
                 win_length=512, augs_enable=False, is_train=False):
        super().__init__()

        self.final_df = self.get_files_from_manifest(train_manifest)
        self.root_audio_folder = Path(train_manifest).parent.absolute().as_posix()
        self.final_df = self.final_df[(self.final_df['duration'] > min_len) & (self.final_df['duration'] < max_len)]
        if file_num > 0:
            self.final_df = self.final_df.head(file_num)
        print(f"Dataset contain {self.final_df['duration'].sum() / 3600:.2f} hours")
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sampling_rate = 16000

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        self.is_train = is_train

        # Augs
        self.augs_enable = augs_enable
        self.min_augs = 0
        self.max_augs = 2
        self.rir_folder = None
        self.noise_files = None

    def set_augmentations(self, augs_enable=False, rir_folder=None, noise_folder="", min_augs=0, max_augs=2):
        self.augs_enable = augs_enable
        self.min_augs = min_augs
        self.max_augs = max_augs
        assert self.min_augs >= 0, "Min augs should be >= 0"
        assert self.max_augs >= 0, "Max augs should be >= 0"
        self.rir_folder = rir_folder
        if noise_folder:
            audio_formats = ["mp3", "wav", "flac"]
            self.noise_files = []
            for format in audio_formats:
                self.noise_files.extend(list(Path(noise_folder).absolute().rglob(f"*{format}")))
        print(f"Augmentations add: {self.augs_enable}")

    @staticmethod
    def get_files_from_manifest(manifest):
        df = pd.read_json(manifest, lines=True)
        df = df[["audio_filepath", "text", "duration"]]

        return df

    def explicit_shuffle(self):
        print("Shuffle dataset")
        self.final_df = self.final_df.sample(frac=1, random_state=time.time_ns() % 2 ** 32).reset_index(drop=True)

    def get_audio_lens(self):
        return self.final_df["duration"].to_list()

    def add_augmentations(self, audio, sample_rate=16000):
        augmentations = []
        np.random.seed(int(time.time()))
        num_augmentations = np.random.randint(self.min_augs, self.max_augs)

        if num_augmentations > 0:
            # EQ
            if np.random.rand() < 0.4 and num_augmentations > 0:
                choise = random.choice(["eq", "low-high"])
                if choise == "eq":
                    augmentations.append(SevenBandParametricEQ(-9, 9, p=1))
                    num_augmentations -= 1
                else:
                    # High-pass filter
                    if np.random.rand() < 0.1 and num_augmentations > 0:
                        min_cutoff_freq = 300
                        max_cutoff_freq = 900
                        augmentations.append(HighPassFilter(min_cutoff_freq=min_cutoff_freq,
                                                            max_cutoff_freq=max_cutoff_freq,
                                                            p=1))
                        num_augmentations -= 1

                    # Low-pass filter
                    if np.random.rand() < 0.1 and num_augmentations > 0:
                        min_cutoff_freq = 1800
                        max_cutoff_freq = 3200
                        augmentations.append(LowPassFilter(min_cutoff_freq=min_cutoff_freq,
                                                           max_cutoff_freq=max_cutoff_freq,
                                                           p=1))
                        num_augmentations -= 1

            # Gain
            if np.random.rand() < 0.3 and num_augmentations > 0:
                augmentations.append(Gain(p=1))
                num_augmentations -= 1

            # Mp3Compression
            if np.random.rand() < 0.3 and num_augmentations > 0:
                augmentations.append(Mp3Compression(
                    min_bitrate=16,
                    max_bitrate=96,
                    backend="fast-mp3-augment",
                    preserve_delay=False,
                    p=1.0
                ))
                num_augmentations -= 1

            # Pitch shift
            if np.random.rand() < 0.18 and num_augmentations > 0:
                min_semitones_rand = -4
                max_semitones_rand = 4
                augmentations.append(PitchShift(min_semitones=min_semitones_rand,
                                                max_semitones=max_semitones_rand,
                                                p=1))
                num_augmentations -= 1

            # Loudness Normalization
            if np.random.rand() < 0.3 and num_augmentations > 0:
                augmentations.append(LoudnessNormalization(p=1))
                num_augmentations -= 1

        augs = Compose(augmentations)

        audio = augs(audio, sample_rate=sample_rate)
        audio = self.add_noise_and_reverb(audio, sample_rate)

        return audio

    def add_noise_and_reverb(self, audio, sample_rate=16000, add_gauss=True):
        if random.random() < 0.15 and self.rir_folder is not None:
            transform_impulse_response = ApplyImpulseResponse(
                ir_path=self.rir_folder, p=1)

            audio_reverbed = transform_impulse_response(audio, sample_rate=sample_rate)
            random_const = np.random.uniform(0.11, 0.35)
            audio_reverbed = audio_reverbed * random_const
            audio = audio_reverbed + (audio * (1 - random_const))

        if random.random() < 0.3 and add_gauss:
            transform_gaussian_noise = AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.005,
                p=1
            )

            audio = transform_gaussian_noise(audio, sample_rate=sample_rate)

        # Gaussian SNR
        if random.random() < 0.2:
            min_snr = np.random.randint(12, 14)
            max_snr = np.random.randint(15, 20)
            gauss_snr = AddGaussianSNR(min_snr_db=min_snr, max_snr_db=max_snr, p=1)
            audio = gauss_snr(audio, sample_rate=sample_rate)

        return audio

    def add_bg_noise(self, audio, min_coef=0.1, max_coef=0.5, p=1.):
        if random.random() < p:
            random_noise = random.choice(self.noise_files)
            noise, sr = librosa.load(random_noise)
            if sr != self.sampling_rate:
                noise = librosa.resample(noise, orig_sr=sr, target_sr=self.sampling_rate)

            final_len = audio.shape[-1]
            if len(noise) < final_len:
                factor = math.ceil(final_len / len(noise))
                noise = np.concat([noise for _ in range(factor)], axis=-1)[:final_len]
            else:
                start = random.randint(0, len(noise) - final_len)
                noise = noise[start: start + final_len]

            noise = noise * random.uniform(min_coef, max_coef)
            noise = noise[:audio.shape[-1]]
            audio = noise + audio

        return audio

    def __getitem__(self, idx):
        file = self.final_df.iloc[idx]
        audio_file_path = os.path.join(self.root_audio_folder, file["audio_filepath"])
        audio, sr = librosa.load(audio_file_path)
        if sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)

        if self.noise_files:
            audio = self.add_bg_noise(audio, min_coef=0.2, max_coef=0.45, p=0.7)

        if self.is_train and self.augs_enable:
            audio = self.add_augmentations(audio, self.sampling_rate)

        encoded_ids = self.tokenizer.encode(file["text"], out_type=int)
        encoded_ids = [el for el in encoded_ids if
                       el != 0]  # Drop unknown tokens for target, because models must not predict it

        audio = self.stft.transform(torch.from_numpy(audio).float().unsqueeze(0))
        audio = audio.transpose(1, 3)

        return audio[0], torch.tensor(encoded_ids)

    def __len__(self):
        return len(self.final_df)


class BucketingSampler(data.Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, bucket_size=400):
        super().__init__()

        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size

        # Index sort
        self.indices = np.argsort(self.lengths)

        # Bucketing
        self.buckets = [
            self.indices[i:i + bucket_size]
            for i in range(0, len(self.indices), bucket_size)
        ]

    def __iter__(self):
        # Inter-Buckets shuffle
        if self.shuffle:
            np.random.shuffle(self.buckets)

        for bucket in self.buckets:
            # Intra-Buckets shuffle
            if self.shuffle:
                np.random.shuffle(bucket)

            # Create batch
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.buckets
        )


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
        padded_item = torch.cat((audio, torch.zeros(audio.shape[0], time_padding, audio.shape[-1], )), dim=1)
        audio_padding_len.append(audio.shape[-2])
        padded_audio.append(padded_item)

        ids_padding = int(max_ids_len - ids.shape[-1])
        padded_item = torch.cat((ids, torch.full(size=(ids_padding,), fill_value=BLANK_TOKEN_ID)), dim=0)
        ids_len.append(ids.shape[-1])
        padded_ids.append(padded_item)

    # Stack padded sequences and return lengths
    return torch.stack(padded_audio), torch.tensor(audio_padding_len), torch.stack(padded_ids), torch.tensor(ids_len)
