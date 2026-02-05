import torch
import torch.nn as nn
import math

from models.common_modules.regularization import DropPath
from models.common_modules.transformer_modules import SwiGLUFFN, GQASelfAttentionRelPos
from models.conformer.conv_module import ConvModule
from models.common_modules.mamba import Mamba2, Mamba2Config
from utils.masking import get_mask_vector, cut_masks_with_len_vector, fix_full_masked_lines

# from conv_module import ConvModule
# from minimal_mamba import Mamba2, Mamba2Config


GLOBAL_EPS = 1e-4


class BaseEncoderBlock(nn.Module):
    def __init__(self, type: str, d_model, n_heads, n_groups, chunk_size, left_context, right_context=0, dropout=0.):
        super().__init__()
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context
        self.n_heads = n_heads
        assert type in ['attn', 'mamba'], "Type must be 'attn' or 'mamba'"
        self.type = type
        max_rel_pos = math.ceil((self.chunk_size + self.left_context + self.right_context) / self.chunk_size) * self.chunk_size

        self.pre_attn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        if self.type == 'attn':
            self.block = GQASelfAttentionRelPos(d_model, num_heads=n_heads, num_groups=n_groups, max_position=max_rel_pos, bias=False, dropout=dropout, )
        elif self.type == 'mamba':
            self.block = Mamba2(Mamba2Config(d_model=d_model, chunk_size=chunk_size, expand=1))

        self.pre_conv_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.conv_block = ConvModule(d_model, 9, dropout=dropout)

        self.pre_ffn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.ffn = SwiGLUFFN(d_model, factor=2)

        self.drop_block = DropPath(dropout)

    def forward(self, x, attn_mask):
        B, L, D = x.shape

        residual = x
        x = self.pre_attn_norm(x)
        if self.type == 'attn':
            padded_x = torch.cat((torch.zeros((B, self.left_context, D), device=x.device, dtype=x.dtype), x), dim=1)
            x = self.block(padded_x, attn_mask=attn_mask)[:, self.left_context:, :]
        else:
            pad_len = (x.shape[1] + self.chunk_size-1) // self.chunk_size * self.chunk_size - x.shape[1]
            padded_x = torch.cat((x, torch.zeros((B, pad_len, D), device=x.device, dtype=x.dtype)), dim=1)
            x = self.block(padded_x)[0][:, :L, :]
        x = residual + self.drop_block(x)

        residual = x
        x = self.pre_conv_norm(x)
        x = residual + self.drop_block(self.conv_block(x))

        residual = x
        x = self.pre_ffn_norm(x)
        x = residual + self.drop_block(self.ffn(x))

        return x


class AudioEncoder(nn.Module):
    def __init__(self, inter_d_model, chunk_size, left_context_chunk_number, right_context_chunk_number, n_heads, n_groups, layer_num=1, mamba_every_n_block=3, dropout=0.1):
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
        self.blocks = nn.ModuleList()
        for i in range(layer_num):
            block_type = "mamba" if (i+1) % mamba_every_n_block == 0 and use_mamba else "attn"
            self.blocks.append(BaseEncoderBlock(block_type, inter_d_model, n_heads, n_groups, self.chunk_size, self.left_context, self.right_context, dropout))

    def forward(self, x, audio_len=None):
        mask = get_mask_vector(self.chunk_size, x.shape[1] + self.left_context, self.left_context, self.right_context, device=x.device)

        if audio_len is not None:
            masks = cut_masks_with_len_vector(mask, audio_len, self.left_context)
        else:
            masks = mask.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, L, L)
        masks = fix_full_masked_lines(masks)
        masks = torch.repeat_interleave(masks, self.n_heads, 0)

        for module in self.blocks:
            x = module(x, masks)

        return x


def count_parameters(model):
    parametrs_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parametrs_num} ({parametrs_num / (10 ** 6):.1f}M)")


if __name__ == "__main__":
    inter_d_model = 512
    chunk_size = 80
    n_heads = 8
    layer_num = 4
    dropout = 0.1
    model = AudioEncoder(inter_d_model, chunk_size, 1, n_heads, layer_num, dropout).eval()
    count_parameters(model)

    x = torch.randn(3, 400, inter_d_model)
    audio_len = torch.tensor([36, 400, 253])
    print(x.shape)
    full_out = model(x, audio_len)
    print(full_out.shape)

    # mask = get_mask_cycle(80, 270, 120)
    # vectorized_mask = get_mask_vector(80, 270, 120, "cpu")
    # print("Masks are same: ", torch.allclose(mask, vectorized_mask, atol=1e-8))
