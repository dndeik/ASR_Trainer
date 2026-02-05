import math
import random

import torch
from torch import nn as nn


class SpecAugment(nn.Module):
    """
    SpecAugment for log-mel / log-spectrogram.
    Input: [B, L, D]
    """

    def __init__(
            self,
            time_mask_percent: float = 0.2,  # fraction of time to mask
            freq_mask_percent: float = 0.2,  # fraction of freq to mask
            max_time_mask: int = 40,
            max_freq_mask: int = 15,
            p: float = 1.0,
            adaptive_time_mask: bool = False,
    ):
        super().__init__()
        self.time_mask_percent = time_mask_percent
        self.freq_mask_percent = freq_mask_percent
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.p = p
        self.adaptive_time_mask = adaptive_time_mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        Args:
            x: [B, L, D]
            lengths: [B] valid lengths (optional)
        """
        if not self.training or random.random() > self.p:
            return x

        B, L, D = x.shape

        for b in range(B):
            valid_L = lengths[b].item() if lengths is not None else L

            # ===== Time masking =====
            total_time_to_mask = int(self.time_mask_percent * valid_L)
            if total_time_to_mask > 0:
                max_t = (
                    int(valid_L * 0.1)
                    if self.adaptive_time_mask
                    else min(self.max_time_mask, valid_L)
                )
                num_time_masks = max(1, math.ceil(total_time_to_mask / max_t))

                for _ in range(num_time_masks):
                    t = random.randint(0, max_t)
                    if t == 0:
                        continue
                    t0 = random.randint(0, max(0, valid_L - t))
                    x[b, t0: t0 + t, :] = 0.0

            # ===== Frequency masking =====
            total_freq_to_mask = int(self.freq_mask_percent * D)
            if total_freq_to_mask > 0:
                max_f = min(self.max_freq_mask, D)
                num_freq_masks = max(1, math.ceil(total_freq_to_mask / max_f))

                for _ in range(num_freq_masks):
                    f = random.randint(0, max_f)
                    if f == 0:
                        continue
                    f0 = random.randint(0, max(0, D - f))
                    x[b, :, f0: f0 + f] = 0.0

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob).float()
        return x / keep_prob * random_tensor
