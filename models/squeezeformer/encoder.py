import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.common_modules.regularization import DropPath
from models.common_modules.transformer_modules import SwiGLUFFN, GQASelfAttentionRelPos
from models.squeezeformer.conv_module import ConvModule, CausalConv1d
from models.common_modules.mamba import Mamba2, Mamba2Config
from utils.masking import get_mask_vector, cut_masks_with_len_vector, fix_full_masked_lines

# from conv_module import ConvModule, CausalConv1d
# from mamba import Mamba2, Mamba2Config

GLOBAL_EPS = 1e-5


class Scaler(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, d_model))
        self.bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        return self.weight * x + self.bias


class BaseEncoderBlock(nn.Module):
    def __init__(self, type: str, d_model, n_heads, n_groups, chunk_size, left_context, right_context=0, conv_kernel=9,
                 dropout=0.):
        super().__init__()
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context
        self.n_heads = n_heads
        assert type in ['attn', 'mamba'], "Type must be 'attn' or 'mamba'"
        self.type = type
        max_rel_pos = math.ceil(
            (self.chunk_size + self.left_context + self.right_context) / self.chunk_size) * self.chunk_size

        if self.type == 'attn':
            self.block = GQASelfAttentionRelPos(d_model, num_heads=n_heads, num_groups=n_groups,
                                                max_position=max_rel_pos, bias=False, dropout=dropout, )
        elif self.type == 'mamba':
            self.block = Mamba2(Mamba2Config(d_model=d_model, chunk_size=chunk_size, expand=1))
        self.attn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.attn_coef = nn.Parameter(torch.ones(1))

        self.first_ffn = SwiGLUFFN(d_model, factor=2)
        self.first_ffn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.first_ffn_coef = nn.Parameter(torch.ones(1))

        self.conv_block = ConvModule(d_model, conv_kernel, dropout=dropout)
        self.conv_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.conv_coef = nn.Parameter(torch.ones(1))

        self.sec_ffn = SwiGLUFFN(d_model, factor=2)
        self.sec_ffn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.sec_ffn_coef = nn.Parameter(torch.ones(1))

        self.drop_block = DropPath(dropout)

    def forward(self, x, attn_mask):
        B, L, D = x.shape

        residual = x
        x = self.first_ffn_norm(x)
        x = residual + self.drop_block(self.first_ffn_coef * self.first_ffn(x))

        residual = x
        x = self.attn_norm(x)
        if self.type == 'attn':
            padded_x = torch.cat((torch.zeros((B, self.left_context, D), device=x.device, dtype=x.dtype), x), dim=1)
            x = self.block(padded_x, attn_mask=attn_mask)[:, self.left_context:, :]
        else:
            pad_len = (x.shape[1] + self.chunk_size - 1) // self.chunk_size * self.chunk_size - x.shape[1]
            padded_x = torch.cat((x, torch.zeros((B, pad_len, D), device=x.device, dtype=x.dtype)), dim=1)
            x = self.block(padded_x)[0][:, :L, :]
        x = residual + self.drop_block(self.attn_coef * x)

        residual = x
        x = self.conv_norm(x)
        x = residual + self.drop_block(self.conv_coef * self.conv_block(x))

        residual = x
        x = self.sec_ffn_norm(x)
        x = x + self.drop_block(self.sec_ffn_coef * self.sec_ffn(x))

        return x


class AudioEncoder(nn.Module):
    def __init__(self, inter_d_model, chunk_size, left_context_chunk_number, right_context_chunk_number, n_heads,
                 n_groups, conv_kernel=9, layer_num=1, mamba_every_n_block=3, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.left_context = int(left_context_chunk_number * chunk_size)
        self.right_context = int(right_context_chunk_number * chunk_size)

        assert mamba_every_n_block != 0, "'mamba_every_n_block' can't be zero"
        if mamba_every_n_block < 0:
            print("'mamba_every_n_block' is less then zero, only GQA block will be used")
        use_mamba = mamba_every_n_block > 0  # if "mamba_every_n_block" < 0 we use only GQA

        first_downscaled_block = layer_num // 2 - 1
        last_downscaled_block = layer_num - 2
        self.downsample_factor = 2

        self.start_full_blocks = nn.ModuleList()
        self.downsampled_blocks = nn.ModuleList()
        self.end_full_blocks = nn.ModuleList()
        for i in range(layer_num):
            block_type = "mamba" if (i + 1) % mamba_every_n_block == 0 and use_mamba else "attn"

            if i >= first_downscaled_block and i <= last_downscaled_block:
                self.downsampled_blocks.append(BaseEncoderBlock(block_type, inter_d_model, n_heads, n_groups,
                                                                self.chunk_size // self.downsample_factor,
                                                                self.left_context // self.downsample_factor,
                                                                self.right_context // self.downsample_factor,
                                                                conv_kernel // self.downsample_factor + conv_kernel % 2,
                                                                dropout))
            elif i < first_downscaled_block:
                self.start_full_blocks.append(
                    BaseEncoderBlock(block_type, inter_d_model, n_heads, n_groups, self.chunk_size, self.left_context,
                                     self.right_context, conv_kernel, dropout))
            else:
                self.end_full_blocks.append(
                    BaseEncoderBlock(block_type, inter_d_model, n_heads, n_groups, self.chunk_size, self.left_context,
                                     self.right_context, conv_kernel, dropout))

        self.downsample_conv = CausalConv1d(inter_d_model, inter_d_model, self.downsample_factor + 1,
                                            stride=self.downsample_factor)

    def _get_finished_mask(self, x, audio_len=None, downsample_factor=1):
        B, L, D = x.shape
        mask = get_mask_vector(self.chunk_size // downsample_factor,
                               (L + self.left_context) // downsample_factor,
                               self.left_context // downsample_factor,
                               self.right_context // downsample_factor,
                               device=x.device)

        if audio_len is not None:
            audio_len = audio_len // downsample_factor
            masks = cut_masks_with_len_vector(mask, audio_len, self.left_context)
        else:
            masks = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        masks = fix_full_masked_lines(masks)
        masks = torch.repeat_interleave(masks, self.n_heads, 0)
        return masks

    def forward(self, x, audio_len=None):
        full_blocks_masks = self._get_finished_mask(x, audio_len, 1)
        downsampled_blocks_masks = self._get_finished_mask(x, audio_len, self.downsample_factor)

        for module in self.start_full_blocks:
            x = module(x, full_blocks_masks)

        residual = x
        src_len = x.shape[1]
        x = x.transpose(1, 2)
        x = self.downsample_conv(x)
        x = x.transpose(1, 2)
        for module in self.downsampled_blocks:
            x = module(x, downsampled_blocks_masks)

        x = x.transpose(1, 2)
        x = F.interpolate(x, size=src_len, mode="nearest")
        x = x.transpose(1, 2)

        x = residual + x

        for module in self.end_full_blocks:
            x = module(x, full_blocks_masks)

        return x


def count_parameters(model):
    parametrs_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parametrs_num} ({parametrs_num / (10 ** 6):.1f}M)")


if __name__ == "__main__":
    inter_d_model = 512
    chunk_size = 80
    n_heads = 8
    layer_num = 10
    dropout = 0.1
    model = AudioEncoder(inter_d_model, chunk_size, 3, 0.5, n_heads, n_heads // 2, 9, layer_num, -1, dropout).eval()
    count_parameters(model)

    x = torch.randn(3, 400, inter_d_model)
    audio_len = torch.tensor([36, 400, 253])
    print(x.shape)
    full_out = model(x, audio_len)
    print(full_out.shape)

    # mask = get_mask_cycle(80, 270, 120)
    # vectorized_mask = get_mask_vector(80, 270, 120, "cpu")
    # print("Masks are same: ", torch.allclose(mask, vectorized_mask, atol=1e-8))
