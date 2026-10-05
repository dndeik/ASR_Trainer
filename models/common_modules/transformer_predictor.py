import torch
import torch.nn as nn

from .transformer_modules import GQASelfAttentionRelPos, SimpleFFN


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalized GQA and feed-forward layers."""

    def __init__(self, hidden_dim, num_heads, num_groups, dropout=0.0, max_position=64):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = GQASelfAttentionRelPos(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            max_position=max_position,
            dropout=dropout,
        )
        self.pre_ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = SimpleFFN(hidden_dim, factor=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(self.pre_attn_norm(x), attn_mask=attn_mask))
        x = x + self.dropout(self.ffn(self.pre_ffn_norm(x)))
        return x


class TransformerDecoder(nn.Module):
    """Causal windowed Transformer predictor for the RNNT head."""

    def __init__(self, num_vocab, embed_dim, hidden_dim, layer_num, dropout=0.0,
                 num_heads=4, num_groups=4, context_size=32, max_position=64):
        super().__init__()
        assert max_position >= context_size

        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.num_heads = num_heads
        self.embedding = nn.Embedding(num_vocab, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, num_groups, dropout, max_position)
            for _ in range(layer_num)
        ])

    def _make_combined_mask(self, y_lengths, T, device):
        B = y_lengths.size(0)
        positions = torch.arange(T, device=device)
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        window_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-self.context_size)
        combined = (causal_mask | window_mask).unsqueeze(0).expand(B, -1, -1).clone()

        key_padding = positions.unsqueeze(0) >= y_lengths.unsqueeze(1)
        combined = combined | key_padding.unsqueeze(1)
        query_padding = positions.unsqueeze(0) >= y_lengths.unsqueeze(1)
        diag = torch.eye(T, device=device, dtype=torch.bool)
        combined = combined | (query_padding.unsqueeze(2) & ~diag.unsqueeze(0))
        combined = combined & ~diag.unsqueeze(0)
        return combined.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, T, T)

    def forward(self, y, y_lengths):
        _, T = y.shape
        x = self.input_proj(self.embedding(y))
        attn_mask = self._make_combined_mask(y_lengths, T, y.device)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x

    def infer(self, y, hidden=None):
        _, T_new = y.shape
        emb = self.input_proj(self.embedding(y))
        full_emb = emb if hidden is None else torch.cat([hidden, emb], dim=1)
        if full_emb.size(1) > self.context_size:
            full_emb = full_emb[:, -self.context_size:]

        x = full_emb
        for layer in self.layers:
            x = layer(x, attn_mask=None)
        return x[:, -T_new:], full_emb.detach()
