import os
import json
from pathlib import Path
import random

import torch
import librosa
import sentencepiece as spm
import pandas as pd
import toml

from torch_stft.stft_implementation.stft import STFT
from models.conformer_plus_mamba.model import ConformerHybrid

# Load dataset
manifest_path = Path("/opt/datasets/ASR_data/CAPITALIZED_TEST_FINAL.jsonl")
df = pd.read_json(manifest_path, lines=True)
df = df[["audio_filepath", "text"]]

sample_num = random.randint(0, len(df))
ref_text = df["text"][sample_num]
audio_path = manifest_path.parent.joinpath(df["audio_filepath"][sample_num])
print("RANDOM SAMPLE NUM: ", sample_num)

# Load audio
# audio_path = "/opt/datasets/ASR_data/Mozila-cv-corpus-18.0-2024-06-14/ru/clips/common_voice_ru_18849869.mp3"
audio, sr = librosa.load(audio_path)

sampling_rate = 16000
if sr != sampling_rate:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
print("AUDIO DUR: ", audio.shape[-1]/sampling_rate)

# Tokenizer
tokenizer = spm.SentencePieceProcessor(model_file=os.path.join("models/ru_tokenizer/tokenizer_spe_bpe_cp", "tokenizer.model"))

# Model
config_path = "cfg_train.toml"
config = json.loads(json.dumps(toml.load(config_path)))
checkpoint_path = "experiments/gqa_with_mamba_BIG_model_NO_BIAS_2025-11-24-23h22m/checkpoints/model_0005.tar"

stft = STFT(config["FFT"]["n_fft"], config["FFT"]["hop_length"])

model = ConformerHybrid(num_vocab=tokenizer.vocab_size()+1,
                        inter_d_model=config['model']['inter_d_model'],
                        n_mel=config['model']['n_mel'],
                        time_factor=config['model']['time_factor'],
                        chunk_size=config['model']['chunk_size'],
                        context_chunk_number=config['model']['context_chunk_number'],
                        freq_dim=config['FFT']['hop_length']+1,
                        n_heads=config['model']['n_heads'],
                        n_groups=config['model']['n_groups'],
                        layer_num=config['model']['layer_num'],
                        mamba_every_n_block=config['model']['mamba_every_n_block'],
                        dropout=0.1)
model.eval()

checkpoint = torch.load(checkpoint_path, weights_only=False)["model"]
model.load_state_dict(checkpoint)

# Infer
audio = stft.transform(torch.from_numpy(audio).unsqueeze(0))
audio = audio.transpose(1, 3)
res = model.infer_rnnt(audio)
print("REF: ", ref_text)
print("EST: ", tokenizer.decode(res))
