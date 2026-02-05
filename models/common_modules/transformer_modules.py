import math

import torch
from torch import nn as nn
from torch.nn import functional as F


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

        q_scaled = q * math.sqrt(1.0 / self.head_dim)

        attn_output_weights = torch.bmm(q_scaled, k_t)

        if attn_mask is not None:
            attn_output_weights = torch.masked_fill(attn_output_weights, attn_mask, float("-inf"))

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
            attn_output_weights = torch.masked_fill(attn_output_weights, attn_mask, float("-inf"))

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


class SimpleFFN(nn.Module):
    def __init__(self, d_model, factor=4) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, factor * d_model, bias=False),
                                 nn.GELU(),
                                 nn.Linear(factor * d_model, d_model, bias=False))

    def forward(self, x):
        return self.ffn(x)


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