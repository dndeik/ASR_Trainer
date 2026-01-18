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

class ConvModule(nn.Module):
    def __init__(self,  d_model, k_size=7, dropout=0.):
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
        x = x.transpose(1, 2)

        return x