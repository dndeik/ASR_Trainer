# Torch STFT

## Description

This repository is inspired by this work. A Short-time Fourier transform is implemented here, which can be converted to ONNX without any problems. 
Now there may be some bugs, but the main functionality works. There is also a wrapper for using real-time models.

## Example

You can find a similar example in `main.py`

```py
from stft_wrapper import DummyModel, STFTStreamWrapper
import torch

n_fft = 320
hop = n_fft // 2
src_buffer = torch.arange(n_fft*10, dtype=torch.float32)

dummy_neural_net = DummyModel()

stft_wrapper = STFTStreamWrapper(dummy_neural_net, n_fft=n_fft, hop_length=hop, window_type="hann")
stft_wrapper.eval()

samples_buffer = torch.zeros((1, n_fft))
fourier_buffer = torch.zeros((1, hop + 1, 2, 2), dtype=torch.float32)
dummy_model_cashe = torch.zeros((1, 100), dtype=torch.float32)

samples_collect = []
for i in range(0, src_buffer.size()[0], hop):
    new_samples = src_buffer[i:i + hop]
    ready_samples, samples_buffer, fourier_buffer, dummy_model_cashe = stft_wrapper(new_samples, samples_buffer, fourier_buffer, dummy_model_cashe)
    samples_collect.append(ready_samples)

output_stream_buffer = torch.cat(samples_collect, dim=0)[stft_wrapper.hop_length:]
```
