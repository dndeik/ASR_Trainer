# ASR Trainer (with models)

## Overview
While working on ASR tasks using Nvidia Nemo, I found a lot of inconvenient things,
starting with inconvenient model architecture changes and ending with complex inheritance and evaluation
of internal parts, so I decided to write my trainer in pure Pytorch. And at the same time implement
several basic models.

## Models
A hybrid (uses both RNNT and CTC heads at once) Conformer-like model with several features has now been implemented:
- MHA has been replaced by GQA (in default, the number of groups = 1/2 of the number of heads)
- FFN uses SwiGLU instead of the basic Linear -> activation -> Linear
- Every third layer uses Mamba2 instead of GQA
- For easier future deployment (and conversion to ONNX) I've excluded kaldi-features. Therefore, the preprocessing
looks like this: raw audio -> STFT -> magnitude -> trainable mel(linear layer inited as mel bank)

The model was designed for use in the stream, so all the blocks are completely causal. So the results of the model on
the train will be completely identical to the results in the stream.

### Metrics
This model with the following parameters (12 layers, dimension 768, chunk of 100 (1 second) and the same context),
trained on ~2300 hours of data, shows the `CommonVoiceRU` test **~1.9%** WER.

### TODO
- [ ] Add trained weights
- [ ] Add a code to convert and compare full model and stream

## Prerequisites
To use the trainer, you need to install Pytorch:
```console
pip3 install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Install `requirements.txt`
```console
pip3 install -r requirements.txt
```
### Additional
If you want to use `ClearML', install its package, configure credentials for it and set it in `cfg_train.toml`
```toml
[logger]
log_to_clearml = true
```

## Datasets
You can use any dataset for the trainer, it is presented in the `.jsonl` file.

The internal structure of the file should be as follows:
```console
{"audio_filepath":"Mozila-cv-corpus-18.0-2024-06-14\/ru\/clips\/common_voice_ru_18849871.mp3","text":"\u0412\u0430\u0448\u0430 \u0432\u043e\u043b\u044f.","duration":1.776}
{"audio_filepath":"Mozila-cv-corpus-18.0-2024-06-14\/ru\/clips\/common_voice_ru_18849872.mp3","text":"\u0412\u0430\u0448\u0430 \u0432\u043e\u043b\u044f \u0432\u043e\u043b\u044f \u0432\u043e\u043b\u044f.","duration":3.256}
```

Each line contains three key fields: `audio_filepath`, `text` and `duration`.

- `audio_filepath` is the relative path to the file **FROM .JSONL FILE**
- `text` - the text in the record
- `duration` - duration of recording

The path to the datasets is specified in `cfg_train.toml`
```toml
[train_dataset]
train_manifest = "/opt/datasets/ASR_data/CAPITALIZED_TRAIN_FINAL.jsonl" # Put the path to your train dataset
```

## Tokenizer
The path to the tokenizer is specified in `cfg_train.toml`
```toml
[tokenizer]
tokenizer_path = "models/ru_tokenizer/tokenizer_spe_bpe_cp"
```
The trainer uses the SentencePiece tokenizer. You can use a tokenizer for the Russian language. 
`models/ru_tokenizer/tokenizer_spe_bpe_cp` (with capitalization and punctuation),
you need to prepare your own for other languages.

## Run
To run trainer you should specify `cfg_train.toml` and run:
```console
python train.py
```

