import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.conformer_plus_mamba.conv_module import ConvModule
from models.conformer_plus_mamba.mamba import Mamba2, Mamba2Config

import random

# from conv_module import ConvModule
# from minimal_mamba import Mamba2, Mamba2Config


GLOBAL_EPS = 1e-4


def get_mask_cycle(chunk_size, mask_length, left_context=0):
    # if self.mask is None or mask_length != self.mask.size()[1]:
    corrected_mask_length = (mask_length + chunk_size) // chunk_size * chunk_size
    mask = torch.zeros(corrected_mask_length, corrected_mask_length + left_context)
    chunk = torch.ones(chunk_size, chunk_size + left_context)
    for i in range(0, mask_length, chunk_size):
        # print(i, i + chunk_size + left_context)
        mask[i:i + chunk_size, i:i + chunk_size + left_context] = chunk

    mask = mask[:mask_length, left_context:mask_length + left_context]

    # self.mask = ~mask.bool()
    mask = ~mask.bool() * float("-inf")
    mask = torch.nan_to_num(mask, neginf=float("-inf"))
    return mask


def get_mask_vector(chunk_size, mask_length, left_context=0, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Выравниваем размер
    corrected_length = (mask_length + chunk_size - 1) // chunk_size * chunk_size

    # 2. Получаем матрицу индексов [corrected_length, corrected_length + left_context]
    row_idx = torch.arange(corrected_length, device=device).unsqueeze(1)  # [L, 1]
    col_idx = torch.arange(corrected_length + left_context, device=device).unsqueeze(0)  # [1, L+LC]

    # 3. Вычисляем маску: какие позиции могут видеть друг друга
    # Каждая строка принадлежит чанку: [0, chunk_size), [chunk_size, 2*chunk_size), ...
    row_chunk = row_idx // chunk_size  # [L, 1]
    col_chunk = col_idx // chunk_size  # [1, L+LC]

    # 4. Строим базовую маску:
    # attention разрешён, если позиция в колонке принадлежит текущему чанку строки
    # и находится в пределах [start_col, start_col + chunk_size + left_context)
    start_col = row_chunk * chunk_size  # [L, 1]
    chunk_mask = (col_idx >= start_col) & (col_idx < start_col + chunk_size + left_context)  # [L, L+LC]

    # 5. Обрезаем до [mask_length, mask_length]
    chunk_mask = chunk_mask[:mask_length, left_context:left_context + mask_length]  # [mask_length, mask_length]

    # 6. Преобразуем в attention mask: 0.0 где разрешено, -inf где запрещено
    attn_mask = torch.where(chunk_mask, 0.0, float('-inf'))

    return attn_mask


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


class BaseEncoderBlock(nn.Module):
    def __init__(self, type: str, d_model, n_heads, n_groups, chunk_size, left_context, dropout=0.):
        super().__init__()
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.n_heads = n_heads
        assert type in ['attn', 'mamba'], "Type must be 'attn' or 'mamba'"
        self.type = type

        self.pre_attn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        if self.type == 'attn':
            self.block = GQASelfAttentionRelPos(d_model, num_heads=n_heads, num_groups=n_groups, max_position=chunk_size+left_context, bias=False, dropout=dropout,)
        elif self.type == 'mamba':
            self.block = Mamba2(Mamba2Config(d_model=d_model, chunk_size=chunk_size))

        self.pre_conv_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.conv_block = ConvModule(d_model, 9, dropout=dropout)

        self.pre_ffn_norm = nn.LayerNorm(d_model, eps=GLOBAL_EPS)
        self.ffn = SwiGLUFFN(d_model, factor=2)

        self.drop_block = DropPath(dropout)

    def forward(self, x, attn_mask):
        B, L, D = x.shape

        x = self.pre_attn_norm(x)
        if self.type == 'attn':
            padded_x = torch.cat((torch.zeros((B, self.left_context, D), device=x.device, dtype=x.dtype), x), dim=1)
            # x = x + self.dropout(self.block(padded_x, padded_x, padded_x, attn_mask=attn_mask, need_weights=False)[0][
            #                          :, self.left_context:, :])
            x = x + self.drop_block(self.block(padded_x, attn_mask=attn_mask)[:, self.left_context:, :])
        else:
            pad_len = (x.shape[1] + self.chunk_size-1) // self.chunk_size * self.chunk_size - x.shape[1]
            padded_x = torch.cat((x, torch.zeros((B, pad_len, D), device=x.device, dtype=x.dtype)), dim=1)
            x = x + self.drop_block(self.block(padded_x)[0][:, :L, :])

        x = self.pre_conv_norm(x)
        x = x + self.drop_block(self.conv_block(x))

        x = self.pre_ffn_norm(x)
        x = x + self.drop_block(self.ffn(x))

        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, factor=4) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, 2 * factor * d_model, bias=False)
        self.linear_2 = nn.Linear(factor * d_model, d_model, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.linear_1(x)
        x1, x2 = torch.chunk(x, 2, -1)
        hidden = self.silu(x1) * x2
        return self.linear_2(hidden)


class SimpleFFN(nn.Module):
    def __init__(self, d_model, factor=4) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, factor * d_model, bias=False),
                                 nn.GELU(),
                                 nn.Linear(factor * d_model, d_model, bias=False))

    def forward(self, x):
        return self.ffn(x)


class GQASelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

        self.embed_dim = embed_dim
        self.num_query_heads = num_heads
        self.num_kv_heads = num_groups
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_groups

        # Раздельные проекции
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_groups)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        bsz, tgt_len, _ = x.size()

        x = x.transpose(0, 1)
        q = self.q_proj(x).view(tgt_len, bsz*self.num_query_heads, self.head_dim).transpose(0, 1) 
        k_t = self.k_proj(x).view(tgt_len, bsz*self.num_kv_heads, self.head_dim).permute(1, 2, 0) 
        v = self.v_proj(x).view(tgt_len, bsz*self.num_kv_heads, self.head_dim).transpose(0, 1) 

        # Повторяем K/V по kv_repeat_factor вдоль head-дим (bsz, num_q_heads, tgt_len, dim)
        k_t = k_t.repeat_interleave(self.kv_repeat_factor, dim=0)
        v = v.repeat_interleave(self.kv_repeat_factor, dim=0)

        q_scaled = q * math.sqrt(1.0 / float(self.head_dim))
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k_t
            )
        else:
            attn_output_weights = torch.bmm(q_scaled, k_t)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, self.embed_dim).transpose(0, 1)

        return attn_output



class GQASelfAttentionRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, max_position=128, bias=False, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert embed_dim % num_groups == 0
        assert num_heads % num_groups == 0

        self.embed_dim = embed_dim
        self.num_query_heads = num_heads
        self.num_kv_heads = num_groups
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_groups
        self.max_position = max_position

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_groups, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Rel Pos [-max_position+1, +max_position-1]
        self.rel_pos_emb = nn.Embedding(2 * max_position - 1, self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, T, D]
        attn_mask: [B * num_heads, T, T] (опционально)
        """
        bsz, tgt_len, _ = x.size()
        x = x.transpose(0, 1)  # [T, B, D]

        q = self.q_proj(x).view(tgt_len, bsz * self.num_query_heads, self.head_dim).transpose(0, 1)
        k_t = self.k_proj(x).view(tgt_len, bsz * self.num_kv_heads, self.head_dim).permute(1, 2, 0)
        v = self.v_proj(x).view(tgt_len, bsz * self.num_kv_heads, self.head_dim).transpose(0, 1)

        k_t = k_t.repeat_interleave(self.kv_repeat_factor, dim=0)
        v = v.repeat_interleave(self.kv_repeat_factor, dim=0)

        q_scaled = q * math.sqrt(1.0 / self.head_dim)

        attn_output_weights = torch.bmm(q_scaled, k_t)

        # Add rel pos
        rel_pos_bias = self.compute_rel_pos_bias(tgt_len, q.device)  # [T, T, head_dim]
        rel_bias_term = torch.einsum("bhd,thd->bht", q_scaled, rel_pos_bias)  # [B*H, T, T]
        attn_output_weights = attn_output_weights + rel_bias_term

        if attn_mask is not None:
            attn_output_weights = attn_output_weights + attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        return attn_output

    def compute_rel_pos_bias(self, seq_len, device):
        """Создаёт матрицу релятивных позиций и возвращает эмбеддинги [T, T, head_dim]."""
        # Диапазон индексов [-T+1, T-1], сдвигаем чтобы не было отрицательных индексов
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clamped = torch.clamp(distance_mat, -self.max_position + 1, self.max_position - 1)
        distance_mat_clamped = distance_mat_clamped + self.max_position - 1
        # [T, T, D_head]
        rel_pos_emb = self.rel_pos_emb(distance_mat_clamped)
        return rel_pos_emb


class AudioEncoder(nn.Module):
    def __init__(self, inter_d_model, chunk_size, context_chunk_number, n_heads, n_groups, layer_num=1, mamba_every_n_block=3, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.left_context = context_chunk_number * chunk_size

        assert mamba_every_n_block != 0, "'mamba_every_n_block' can't be zero"
        if mamba_every_n_block < 0:
            print("'mamba_every_n_block' is less then zero, only GQA block will be used")

        use_mamba = mamba_every_n_block > 0  # if "mamba_every_n_block" < 0 we use only GQA
        self.blocks = nn.ModuleList()
        for i in range(layer_num):
            block_type = "mamba" if (i+1) % mamba_every_n_block == 0 and use_mamba else "attn"
            self.blocks.append(BaseEncoderBlock(block_type, inter_d_model, n_heads, n_groups, self.chunk_size, self.left_context, dropout))

    def _cut_masks_with_len_cycle(self, mask, lens):
        B, L = lens.shape
        masks = []
        for i in range(B):
            cur_masks = torch.full_like(mask, fill_value=-5e4, device=mask.device)
            cur_masks[:lens[i] + self.left_context, :lens[i] + self.left_context] = mask[
                :lens[i] + self.left_context, :lens[i] + self.left_context]
            masks.append(cur_masks)
        masks = torch.stack(masks)
        return masks

    def _cut_masks_with_len_vector(self, mask, lens):
        B = lens.size(0)
        L = mask.size(0)  # размер базовой маски

        # Заполним маски -inf (замаскированные)
        batch_masks = torch.full((B, L, L), fill_value=-5e4, device=mask.device)

        # Индексы для строк и столбцов
        rows = torch.arange(L, device=mask.device).unsqueeze(0).expand(B, -1)  # (B, L)
        cols = torch.arange(L, device=mask.device).unsqueeze(0).expand(B, -1)  # (B, L)

        # Длины с учетом контекста
        max_len = (lens + self.left_context).unsqueeze(-1)  # (B, 1)

        # Булевы маски для строк и столбцов
        row_mask = rows < max_len  # (B, L)
        col_mask = cols < max_len  # (B, L)

        # Пересечение масок для двумерной маски
        valid_mask = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)  # (B, L, L)

        # Расширяем базовую маску на батч
        mask_expanded = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)

        # Формируем итоговую маску с подблоками из базовой маски
        batch_masks = torch.where(valid_mask, mask_expanded, batch_masks)

        return batch_masks

    def forward(self, x, audio_len=None):
        # # GET MASK
        # if random.random() > 0.4 or not self.training:
        #     mask = get_mask_vector(self.chunk_size, x.shape[1] + self.left_context, self.left_context, device=x.device)
        # else:
        #     mask = torch.zeros(x.shape[1] + self.left_context, x.shape[1] + self.left_context).to(x.device)

        mask = get_mask_vector(self.chunk_size, x.shape[1] + self.left_context, self.left_context, device=x.device)

        if audio_len is not None:
            masks = self._cut_masks_with_len_vector(mask, audio_len)
        else:
            masks = mask.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, L, L)
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
