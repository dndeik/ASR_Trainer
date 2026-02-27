import torch
import torch.nn as nn

from stft_implementation.stft import STFT

class STFTStreamWrapper(nn.Module):
    def __init__(self, neural_model, n_fft=320, hop_length=160, window_type="hann"):
        super(STFTStreamWrapper, self).__init__()
        self.neural_model = neural_model
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = STFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window_type
        )

    def forward(self, new_samples, previous_samples_buffer, fourier_buffer, *args, **kwargs):
        """
        Arguments:
            new_samples {tensor} -- New samples with shape (hop * chunk_size)
            previous_samples_buffer {tensor} -- Samples buffer from last iteration with shape (1, hop)
            fourier_buffer {tensor} -- Fourier buffer from last iteration with shape (1, hop_length+1, 1, 2)
            kwargs {dict} -- Additional arguments (cashes) for neural network

        Returns:
            previous_samples {tensor} -- Previous reconstructed audio (data) samples with shape (hop * chunk_size)
            previous_samples_buffer {tensor} -- Samples buffer from last iteration with shape (1, hop)
            fourier_buffer {tensor} -- Fourier buffer from last iteration with shape (1, hop_length+1, 1, 2)
            kwargs {dict} -- Additional arguments (cashes) for neural network
        """
        samples_buffer = torch.cat([previous_samples_buffer, new_samples.unsqueeze(0)], dim=1)

        self.stft.center = True
        packet_stft_res = self.stft.transform(samples_buffer)
        model_estimation = self.neural_model(packet_stft_res, *args, **kwargs)
        res = model_estimation[0]
        fourier_buffer = torch.cat([fourier_buffer, res], dim=2)

        self.stft.center = False
        processed_samples = self.stft.inverse(fourier_buffer)[0]
        previous_samples_buffer = samples_buffer[:, -self.hop_length:]
        fourier_buffer = fourier_buffer[:, :, -1, :].unsqueeze(2)

        return processed_samples, previous_samples_buffer, fourier_buffer, *model_estimation[1:]


class STFTStreamWrapperNoLatency(nn.Module):
    def __init__(self, neural_model, n_fft=320, hop_length=160, window_type="hann"):
        super(STFTStreamWrapperNoLatency, self).__init__()
        self.neural_model = neural_model
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = STFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window_type,
            center=True
        )

    def forward(self, new_samples, samples_buffer, *args, **kwargs):
        """
        Arguments:
            new_samples {tensor} -- New samples with shape (hop)
            samples_buffer {tensor} -- Samples buffer from last iteration with shape (1, n_fft)
            kwargs {dict} -- Additional arguments (cashes) for neural network

        Returns:
            previous_samples {tensor} -- Previous reconstructed audio (data) samples with shape (hop)
            samples_buffer {tensor} -- Samples buffer from last iteration with shape (1, n_fft)
            kwargs {dict} -- Additional arguments (cashes) for neural network
        """
        samples_buffer[0:1, :self.hop_length] = samples_buffer[0:1, self.hop_length:]
        samples_buffer[0:1, self.hop_length:] = new_samples

        packet_stft_res = self.stft.transform(samples_buffer)
        model_estimation = self.neural_model(packet_stft_res, *args, **kwargs)

        processed_samples = self.stft.inverse(model_estimation[0])[0, self.hop_length:]
        return processed_samples, samples_buffer, *model_estimation[1:]



class DummyModel(nn.Module):
    def __init__(self, n_fft=320, hop_length=160):
        super(DummyModel, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x, model_cache):
        model_cache = torch.randn(model_cache.size())
        return x, model_cache


if __name__ == "__main__":
    n_fft = 320
    hop = n_fft // 2
    chunk_size = 8
    src_buffer = torch.rand(n_fft*40, dtype=torch.float32)

    # Casual wrapper
    dummy_neural_net = DummyModel()

    stft_wrapper = STFTStreamWrapper(dummy_neural_net, n_fft=n_fft, hop_length=hop, window_type="hann")
    stft_wrapper.eval()

    samples_buffer = torch.zeros((1, hop))
    fourier_buffer = torch.zeros((1, hop + 1, 1, 2), dtype=torch.float32)

    dummy_model_cashe = torch.zeros((1, 100), dtype=torch.float32)

    samples_collect = []
    for i in range(0, src_buffer.size()[0], chunk_size*hop):
        new_samples = src_buffer[i:i + chunk_size*hop]
        ready_samples, samples_buffer, fourier_buffer, dummy_model_cashe = stft_wrapper(new_samples, samples_buffer, fourier_buffer, dummy_model_cashe)
        samples_collect.append(ready_samples)

    output_stream_buffer = torch.cat(samples_collect, dim=0)[stft_wrapper.hop_length:]

    eps = 0.000001
    print(f"Error in each element < {eps}: {torch.allclose(output_stream_buffer, src_buffer[:-hop], atol=eps)}")

    # # Wrapper with no latency
    # stft_wrapper = STFTStreamWrapperNoLatency(dummy_neural_net, n_fft=n_fft, hop_length=hop, window_type="hann")
    # stft_wrapper.eval()
    #
    # samples_buffer = torch.zeros((1, n_fft))
    # dummy_model_cashe = torch.zeros((1, 100), dtype=torch.float32)
    #
    # samples_collect = []
    # for idx, i in enumerate(range(0, src_buffer.size()[0], hop)):
    #     new_samples = src_buffer[i:i + hop]
    #     ready_samples, samples_buffer, dummy_model_cashe = stft_wrapper(new_samples, samples_buffer, dummy_model_cashe)
    #     print(f"{idx} PACKETS ERRORS: {new_samples - ready_samples}")
    #     samples_collect.append(ready_samples)
    #
    # output_stream_buffer = torch.cat(samples_collect, dim=0)
    # print(output_stream_buffer)
    #
    # eps = 0.001
    # print(f"Error in each element < {eps}: {torch.allclose(output_stream_buffer, src_buffer, atol=eps)}")