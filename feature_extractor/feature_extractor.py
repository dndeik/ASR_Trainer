import torch
import torch.nn as nn
import torchaudio

from .torch_stft.stft_implementation.stft import STFT


class StreamingEMACMVN(nn.Module):
    def __init__(
        self,
        num_features: int,
        alpha: float = 1e-3,
        eps: float = 1e-5,
        chunk_size: int = 32,
        overlap: float = 0.5,
        affine: bool = False,
    ):
        super().__init__()

        assert 0.0 <= overlap < 1.0

        self.num_features = num_features
        self.alpha = alpha
        self.eps = eps

        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap)

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    # ---------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------
    def init_mu_var(self, device=None):
        mu = torch.zeros(self.num_features, device=device)
        var = torch.ones(self.num_features, device=device)
        return mu, var

    def init_overlap(self, batch_size: int, device=None):
        if self.overlap_size == 0:
            return None
        return torch.zeros(
            batch_size, self.overlap_size, self.num_features, device=device
        )

    # ---------------------------------------------------------
    # INFERENCE / STREAMING (single chunk)
    # ---------------------------------------------------------
    @torch.no_grad()
    def infer_chunk(
        self,
        x_chunk: torch.Tensor,          # [B, C, F]
        mu: torch.Tensor,               # [F]
        var: torch.Tensor,              # [F]
        overlap: torch.Tensor | None,   # [B, O, F] or None
    ):
        """
        Returns:
            x_norm: [B, C, F]
            new_mu: [F]
            new_var: [F]
            new_overlap: [B, O, F] or None
        """
        B, C, F = x_chunk.shape
        assert C == self.chunk_size
        assert F == self.num_features

        # ----- statistics window -----
        if self.overlap_size > 0:
            stat_chunk = torch.cat([overlap, x_chunk], dim=1)  # [B, O+C, F]
        else:
            stat_chunk = x_chunk

        chunk_mean = stat_chunk.mean(dim=(0, 1))
        chunk_var = ((stat_chunk - mu) ** 2).mean(dim=(0, 1))

        new_mu = (1.0 - self.alpha) * mu + self.alpha * chunk_mean
        new_var = (1.0 - self.alpha) * var + self.alpha * chunk_var

        # ----- normalize FULL chunk -----
        x_norm = (x_chunk - new_mu) / torch.sqrt(new_var + self.eps)
        if self.gamma is not None:
            x_norm = self.gamma * x_norm + self.beta

        # ----- update overlap -----
        if self.overlap_size > 0:
            new_overlap = x_chunk[:, -self.overlap_size :, :]
        else:
            new_overlap = None

        return x_norm, new_mu, new_var, new_overlap

    # ---------------------------------------------------------
    # TRAIN / BATCH-SAFE (simulated streaming)
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x: [B, T, F]
        """
        B, T, F = x.shape
        assert F == self.num_features

        device = x.device
        mu, var = self.init_mu_var(device)
        overlap = self.init_overlap(B, device)

        out = torch.empty_like(x)

        t = 0
        while t + self.chunk_size <= T:
            x_chunk = x[:, t:t + self.chunk_size, :]
            x_norm, mu, var, overlap = self.infer_chunk(
                x_chunk, mu, var, overlap
            )
            out[:, t:t + self.chunk_size, :] = x_norm
            t += self.chunk_size

        # хвост (опционально)
        if t < T:
            tail = x[:, t:, :]
            pad = self.chunk_size - tail.size(1)
            tail = torch.nn.functional.pad(tail, (0, 0, 0, pad))
            x_norm, _, _, _ = self.infer_chunk(tail, mu, var, overlap)
            out[:, t:, :] = x_norm[:, : T - t, :]

        return out


class FeaturesExractor(nn.Module):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            win_length: int,
            n_mel: int,
            eps: float = 1e-6,
            trainable_mel: bool = False,
    ):
        super().__init__()

        self.eps = eps
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = STFT(n_fft, self.hop_length, self.win_length)
        self.mel = self._get_mel(self.hop_length + 1, n_mel, trainable=trainable_mel)

    @staticmethod
    def _get_mel(freq_dim, n_mel, trainable=True):
        mel_fb = torchaudio.functional.melscale_fbanks(n_freqs=freq_dim, f_min=0, f_max=8000, n_mels=n_mel,
                                                       sample_rate=16000)
        linear = nn.Linear(freq_dim, n_mel, bias=False)
        linear.weight.data = mel_fb.T.data

        for p in linear.parameters():
            p.requires_grad = trainable

        return linear

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, audio_len):
        """
        input:
        x [B, L]
        audio_len [B]

        out:
        x [B, T, n_mel]
        audio_len [B]
        """

        x = self.stft.transform(x)
        x = x.transpose(1, 3)
        x = torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)  # mag [B, T, F]

        spec_lens = ((audio_len + 1 * self.hop_length) // self.hop_length)
        spec_lens = torch.clamp(spec_lens, min=0, max=x.shape[1])

        x = self.mel(x)
        x = torch.log(x + self.eps)  # log-mel

        # На всякий обнуляем все что выше длины
        mask = torch.arange(x.size(1), device=x.device)[None, :] < spec_lens[:, None]
        x.masked_fill_(~mask.unsqueeze(-1), 0.0)

        return x, spec_lens


if __name__ == "__main__":
    bs = 1
    f_dim = 161
    chunk_size = 100

    cmvn = StreamingEMACMVN(f_dim, chunk_size=chunk_size, overlap=0.5)

    dummy_input = torch.rand(bs, chunk_size * 10, f_dim)

    # Full
    full_res = cmvn(dummy_input)

    # Stream
    mu, var = cmvn.init_mu_var()
    overlap = cmvn.init_overlap(batch_size=bs)
    print(overlap.shape)
    stream_res = []
    for i in range(0, dummy_input.shape[1], chunk_size):
        cur_chunk = dummy_input[:, i:i + chunk_size, :]
        norm_chunk, mu, var, overlap = cmvn.infer_chunk(
            cur_chunk, mu, var, overlap
        )
        stream_res.append(norm_chunk)

    stream_res = torch.cat(stream_res, dim=1)

    print(torch.allclose(full_res, stream_res, atol=1e-6))   
