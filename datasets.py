import os
import time
import random
import math
import torch
import pandas as pd
import numpy as np
import librosa
from typing import Optional
from pathlib import Path
from torch.utils import data
import torchaudio

from audiomentations import Compose, PitchShift, HighPassFilter, LowPassFilter, AddGaussianSNR, ApplyImpulseResponse, \
    AddGaussianNoise, SevenBandParametricEQ, LoudnessNormalization, Gain, Mp3Compression, TimeStretch


class MyDataset(data.Dataset):
    def __init__(self, tokenizer, blank_token_id: int, train_manifest: str, file_num: int,
                 min_len: float, max_len: float, augs_enable: bool = False, is_train: bool = False,
                 lang_switch_enable: bool = False):
        super().__init__()

        self.final_df = self.get_files_from_manifest(train_manifest)
        self.root_audio_folder = Path(train_manifest).parent.absolute().as_posix()
        self.final_df = self.final_df[(self.final_df['duration'] > min_len) & (self.final_df['duration'] < max_len)]
        if file_num > 0:
            self.final_df = self.final_df.head(file_num)
        print(f"Dataset contain {self.final_df['duration'].sum() / 3600:.2f} hours")
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.blank_token_id = blank_token_id
        self.sampling_rate = 16000

        self.is_train = is_train

        self.augs_enable = augs_enable
        self.min_augs = 0
        self.max_augs = 2
        self.rir_folder = None
        self.noise_files = None

        self.lang_switch_enable = lang_switch_enable
        if self.lang_switch_enable:
            self.lang_to_indices = {}
            self.lang_to_durations = {}

            for lang in self.final_df["lang"].unique():
                idxs = np.where(self.final_df["lang"].values == lang)[0]
                self.lang_to_indices[lang] = idxs
                self.lang_to_durations[lang] = self.final_df["duration"].values[idxs]

    def set_augmentations(self, augs_enable=False, rir_folder=None, noise_folder="", min_augs=0, max_augs=2,
                          lang_switch_enable=False):
        self.augs_enable = augs_enable
        self.lang_switch_enable = lang_switch_enable
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
        df = df[["audio_filepath", "text", "duration", "lang"]]

        return df

    def explicit_shuffle(self):
        print("Shuffle dataset")
        self.final_df = self.final_df.sample(frac=1, random_state=time.time_ns() % 2 ** 32).reset_index(drop=True)

    def get_audio_lens(self):
        return self.final_df["duration"].to_list()

    def add_augmentations(self, audio, sample_rate=16000, add_gauss=False):
        augmentations = []
        np.random.seed(int(time.time()))
        num_augmentations = np.random.randint(self.min_augs, self.max_augs)

        if num_augmentations > 0:
            if np.random.rand() < 0.4 and num_augmentations > 0:
                choise = random.choice(["eq", "low-high"])
                if choise == "eq":
                    augmentations.append(SevenBandParametricEQ(-10, 10, p=1))
                    num_augmentations -= 1
                else:
                    if np.random.rand() < 0.2 and num_augmentations > 0:
                        min_cutoff_freq = 1000
                        max_cutoff_freq = 1900
                        augmentations.append(HighPassFilter(min_cutoff_freq=min_cutoff_freq,
                                                            max_cutoff_freq=max_cutoff_freq,
                                                            p=1))
                        num_augmentations -= 1

                    if np.random.rand() < 0.2 and num_augmentations > 0:
                        min_cutoff_freq = 700
                        max_cutoff_freq = 5600
                        augmentations.append(LowPassFilter(min_cutoff_freq=min_cutoff_freq,
                                                           max_cutoff_freq=max_cutoff_freq,
                                                           p=1))
                        num_augmentations -= 1

            if np.random.rand() < 0.4 and num_augmentations > 0:
                augmentations.append(Gain(p=1))
                num_augmentations -= 1

            if np.random.rand() < 0.4 and num_augmentations > 0:
                augmentations.append(Mp3Compression(
                    min_bitrate=16,
                    max_bitrate=96,
                    backend="fast-mp3-augment",
                    preserve_delay=False,
                    p=1.0
                ))
                num_augmentations -= 1

            if np.random.rand() < 0.2 and num_augmentations > 0:
                min_semitones_rand = -4
                max_semitones_rand = 4
                augmentations.append(PitchShift(min_semitones=min_semitones_rand,
                                                max_semitones=max_semitones_rand,
                                                p=1))
                num_augmentations -= 1

            if np.random.rand() < 0.3 and num_augmentations > 0:
                augmentations.append(LoudnessNormalization(p=1))
                num_augmentations -= 1

        augs = Compose(augmentations)

        audio = augs(audio, sample_rate=sample_rate)
        audio = self.add_noise_and_reverb(audio, sample_rate, add_gauss=add_gauss)

        return audio

    def add_noise_and_reverb(self, audio, sample_rate=16000, add_gauss=True):
        if random.random() < 0.3 and self.rir_folder is not None:
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

        if random.random() < 0.2:
            min_snr = np.random.randint(12, 14)
            max_snr = np.random.randint(15, 20)
            gauss_snr = AddGaussianSNR(min_snr_db=min_snr, max_snr_db=max_snr, p=1)
            audio = gauss_snr(audio, sample_rate=sample_rate)

        return audio

    @staticmethod
    def rms(x, eps=1e-8):
        return np.sqrt(np.mean(x ** 2) + eps)

    def snr_mix(self, target, interferer, snr_db):
        target_rms = self.rms(target)

        if target_rms > 1e-2:
            noise_rms = self.rms(interferer)
            desired_noise_rms = target_rms / (10 ** (snr_db / 20))
            interferer = interferer * (desired_noise_rms / noise_rms)
        else:
            interferer = interferer * random.uniform(0.25, 1.6)

        mixture = target + interferer

        return mixture

    def add_bg_noise(self, audio, p=1.):
        if random.random() < p:
            random_noise = random.choice(self.noise_files)

            noise_wav, sr = torchaudio.load(str(random_noise))
            if sr != self.sampling_rate:
                noise_wav = torchaudio.functional.resample(noise_wav, sr, self.sampling_rate)
            noise = noise_wav.squeeze(0).numpy()

            final_len = audio.shape[-1]
            if len(noise) < final_len:
                factor = math.ceil(final_len / len(noise))
                noise = np.concatenate([noise for _ in range(factor)], axis=-1)[:final_len]
            else:
                start = random.randint(0, len(noise) - final_len)
                noise = noise[start: start + final_len]

            noise = noise[:audio.shape[-1]]
            audio = self.snr_mix(audio, noise, random.uniform(5, 12))

        return audio

    def add_speed_perturbation(self, audio, sample_rate=16000, p=0.5):
        if random.random() < p:
            speed_factor = random.choice([0.9, 1.1])

            max_samples = int(self.max_len * sample_rate)
            predicted_len = len(audio) / speed_factor
            if predicted_len > max_samples:
                return audio

            time_stretch = TimeStretch(
                min_rate=speed_factor,
                max_rate=speed_factor,
                leave_length_unchanged=False,
                p=1.0
            )

            audio = time_stretch(audio, sample_rate=sample_rate)

        return audio

    def __getitem__(self, idx):
        file = self.final_df.iloc[idx]

        audio_path = os.path.join(self.root_audio_folder, file["audio_filepath"])
        audio = self._read_audio(audio_path)

        text = file["text"]
        lang = file["lang"]
        duration = file["duration"]

        token_ids = self.tokenizer.encode(text, out_type=int)
        token_ids = [t for t in token_ids if t != 0]

        # 1. Code-switching
        if self.lang_switch_enable and random.random() < 0.25:
            other_audio, other_tokens, other_lang = self.get_another_lang_sample(lang, duration)

            if other_audio is not None:
                pause_dur = random.uniform(0.2, 0.5)
                pause = np.zeros(int(self.sampling_rate * pause_dur), dtype=audio.dtype)
                audio = np.concatenate([audio, pause, other_audio], axis=-1)

                token_ids.extend(other_tokens)

        # 2. Speed Perturbation
        if self.is_train and self.augs_enable:
            audio = self.add_speed_perturbation(audio, self.sampling_rate, p=0.5)

        # 3. Background Noise
        if self.is_train and self.noise_files and self.augs_enable:
            audio = self.add_bg_noise(audio, p=0.3)

        # 4. Spectral Augmentations (EQ, Pitch, Gain, etc.)
        if self.is_train and self.augs_enable:
            audio = self.add_augmentations(audio, self.sampling_rate, add_gauss=False)

        return torch.from_numpy(audio), torch.tensor(token_ids)

    def _read_audio(self, audio_file_path):
        try:
            waveform, sr = torchaudio.load(audio_file_path)
        except Exception as e:
            print(f"[WARNING] torchaudio failed to load {audio_file_path}: {e}. Falling back to librosa.")
            audio_np, sr = librosa.load(audio_file_path, sr=None)
            waveform = torch.from_numpy(audio_np).unsqueeze(0)

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        return waveform.squeeze(0).numpy()

    def __len__(self):
        return len(self.final_df)

    def custom_collate_fn(self, batch):
        audio_len = []
        ids_len = []
        for audio, ids in batch:
            audio_len.append(audio.shape[0])
            ids_len.append(ids.shape[-1])

        max_audio_len = max(audio_len)
        max_ids_len = max(ids_len)

        audio_len = []
        padded_audio = []

        ids_len = []
        padded_ids = []
        for audio, ids in batch:
            time_padding = max_audio_len - audio.shape[0]
            padded_item = torch.cat((audio, torch.zeros(time_padding)), dim=0)
            audio_len.append(audio.shape[0])
            padded_audio.append(padded_item)

            ids_padding = int(max_ids_len - ids.shape[-1])
            padded_item = torch.cat((ids, torch.full(size=(ids_padding,), fill_value=self.blank_token_id)), dim=0)
            ids_len.append(ids.shape[-1])
            padded_ids.append(padded_item)

        return torch.stack(padded_audio), torch.tensor(audio_len), torch.stack(padded_ids), torch.tensor(ids_len)

    def get_another_lang_sample(self, cur_lang, cur_len):
        max_additional_len = self.max_len - cur_len

        if max_additional_len <= 1.0:
            return None, [], None

        pause_dur = random.uniform(0.2, 0.5)
        budget = max_additional_len - pause_dur

        other_langs = [l for l in self.lang_to_indices.keys() if l != cur_lang]
        other_lang = random.choice(other_langs)

        idxs = self.lang_to_indices[other_lang]
        durs = self.lang_to_durations[other_lang]

        valid = durs <= budget
        if not np.any(valid):
            return None, [], None

        valid_idxs = idxs[valid]
        other_idx = np.random.choice(valid_idxs)
        other_row = self.final_df.iloc[other_idx]

        other_audio_path = os.path.join(self.root_audio_folder, other_row["audio_filepath"])
        other_audio = self._read_audio(other_audio_path)

        other_tokens = self.tokenizer.encode(other_row["text"], out_type=int)
        other_tokens = [t for t in other_tokens if t != 0]

        return other_audio, other_tokens, other_lang


class BucketingSampler(data.Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, bucket_size=400, drop_last=False):
        super().__init__()

        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.drop_last = drop_last

        self.indices = np.argsort(self.lengths)

        self.buckets = [
            self.indices[i:i + bucket_size].copy()
            for i in range(0, len(self.indices), bucket_size)
        ]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.buckets)

        for bucket in self.buckets:
            if self.shuffle:
                np.random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(bucket) // self.batch_size for bucket in self.buckets)
        return sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.buckets
        )