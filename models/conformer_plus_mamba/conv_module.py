import torch
import torch.nn as nn


class TransposedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, bias=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        """
        kernel_size: размер окна вдоль L
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size должен быть нечётным"

        self.conv = nn.Sequential(nn.ConstantPad1d((kernel_size - 1, 0), 0),
                                  nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=0,
                                            bias=False),
                                  )

    def forward(self, x):
        """
        x: (B, C, T)
        """
        # Pooling по D (channels)
        avg_pool = x.mean(dim=1, keepdim=True)  # (B, 1, T)
        max_pool, _ = x.max(dim=1, keepdim=True)  # (B, 1, T)

        # Concat → (B, 2, T)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        attn = torch.sigmoid(self.conv(pooled))  # (B, 1, T)

        # Apply attention
        return x * attn  # (B, C, T)


class ConvModule(nn.Module):
    def __init__(self, d_model, k_size=7, dropout=0.):
        super().__init__()

        self.pre_linear = nn.Conv1d(d_model, d_model, 1)
        inter_d_model = d_model // 2
        self.glu = nn.GLU(1)
        self.conv_block = nn.Sequential(nn.ConstantPad1d((k_size - 1, 0), 0),
                                        nn.Conv1d(inter_d_model, inter_d_model, k_size, groups=inter_d_model),
                                        TransposedLayerNorm(inter_d_model),
                                        nn.SiLU(),
                                        )
        self.post_linear = nn.Conv1d(inter_d_model, d_model, 1)
        self.spatial_attn = SpatialAttention1D(k_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        input: [B, T, C]
        """
        x = x.transpose(1, 2)
        x = self.glu(self.pre_linear(x))
        x = self.conv_block(x)
        x = self.dropout(x)
        x = self.post_linear(x)
        x = self.spatial_attn(x)
        x = x.transpose(1, 2)

        return x