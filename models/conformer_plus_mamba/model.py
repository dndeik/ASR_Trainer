import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from models.conformer_plus_mamba.encoder import AudioEncoder
from models.conformer_plus_mamba.feature_extractor import FeaturesExractor
from utils.decoding import ctc_greedy_decode_batch

# from encoder import AudioEncoder

GLOBAL_EPS = 1e-4


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


class DownsampleConv(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, stride=1):
        super().__init__()

        self.t_pad = k_size - stride
        self.conv_padding = nn.ConstantPad1d((self.t_pad, 0), 0)

        self.conv = nn.Conv1d(in_channel, out_channel, k_size, stride=stride)

    def forward(self, x):
        """[B, T, C]"""
        x = x.transpose(1, 2)
        x = self.conv_padding(x)
        x = self.conv(x)  # [B, C, T//stride]
        x = x.transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, num_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, embed_dim)
        self.predictor = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=1)

    def forward(self, y, y_lengths):
        emb = self.embedding(y)
        y_lengths = y_lengths.cpu()
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb, y_lengths, batch_first=True, enforce_sorted=False)
        packed_seq, _ = self.predictor(packed_seq)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
        return out

    def infer(self, y, hidden=None):
        emb = self.embedding(y)
        out, hidden = self.predictor(emb, hidden)
        return out, hidden


class CTCDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.predictor = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=1)

    def forward(self, x, x_lengths):
        x_lengths = x_lengths.cpu()
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        packed_seq, _ = self.predictor(packed_seq)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
        return out

    def infer(self, x):
        out, _ = self.predictor(x)
        out = self.dropout(out)
        return out


class CTCConv(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, stride=1):
        super().__init__()

        self.t_pad = k_size - stride
        self.conv_padding = nn.ConstantPad1d((self.t_pad, 0), 0)

        self.conv = nn.Conv1d(in_channel, out_channel, k_size, stride=stride)

    def forward(self, x):
        """[B, T, C]"""
        x = x.transpose(1, 2)
        x = self.conv_padding(x)
        x = self.conv(x)  # [B, C, T//stride]
        x = x.transpose(1, 2)
        return x


class Joiner(nn.Module):
    def __init__(self, enc_dim, dec_dim, joint_dim, num_vocab, dropout=0.1):
        super().__init__()
        self.fc_enc = nn.Linear(enc_dim, joint_dim)
        self.fc_pred = nn.Linear(dec_dim, joint_dim)
        self.fc_out = nn.Linear(joint_dim * 2, num_vocab)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, pred_out):
        enc_out = self.dropout(enc_out)
        pred_out = self.dropout(pred_out)

        enc_out = self.fc_enc(enc_out).unsqueeze(2)
        pred_out = self.fc_pred(pred_out).unsqueeze(1)

        T = enc_out.size(1)
        U = pred_out.size(2)

        enc_exp = enc_out.expand(-1, T, U, -1)  # [B, T, U, joint_dim]
        pred_exp = pred_out.expand(-1, T, U, -1)  # [B, T, U, joint_dim]

        joint = torch.cat([enc_exp, pred_exp], dim=-1)  # [B, T, U, joint_dim * 2]
        joint = self.dropout(joint)

        out = self.fc_out(joint)  # [B, T, U, num_vocab]
        return out


class SimpleJoiner(nn.Module):
    def __init__(self, enc_dim, dec_dim, joint_dim, num_vocab, bias=False, dropout=0.1):
        super().__init__()
        self.fc_enc = nn.Linear(enc_dim, joint_dim, bias=bias)
        self.fc_pred = nn.Linear(dec_dim, joint_dim, bias=bias)
        self.fc_out = nn.Linear(joint_dim, num_vocab, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, pred_out):
        enc_out = self.fc_enc(enc_out).unsqueeze(2)
        pred_out = self.fc_pred(pred_out).unsqueeze(1)

        T = enc_out.size(1)
        U = pred_out.size(2)

        enc_exp = enc_out.expand(-1, T, U, -1)  # [B, T, U, joint_dim]
        pred_exp = pred_out.expand(-1, T, U, -1)  # [B, T, U, joint_dim]

        joint = enc_exp + pred_exp  # [B, T, U, joint_dim]
        joint = self.act(joint)
        # joint = self.dropout(joint)

        out = self.fc_out(joint)  # [B, T, U, num_vocab]
        return out


class ConformerHybrid(nn.Module):
    def __init__(self, num_vocab, encoder_d_model,
                 predictor_d_model, joiner_d_model, freq_dim, n_mel, time_factor, n_heads, n_groups, chunk_size,
                 left_context_chunk_number=1, right_context_chunk_number=0,
                 layer_num=8, mamba_every_n_block=3, dropout=0.1):
        super().__init__()
        # Encoder part
        downsampled_chunk_size = chunk_size // time_factor
        self.features_extractor = FeaturesExractor(freq_dim, n_mel, chunk_size=chunk_size, eps=GLOBAL_EPS)
        self.spec_aug = SpecAugment(0.05, 0.06, 5, 6)
        self.downsample_conv = DownsampleConv(n_mel, encoder_d_model, k_size=time_factor * 2 + 1, stride=time_factor)
        self.encoder = AudioEncoder(encoder_d_model, downsampled_chunk_size, left_context_chunk_number, right_context_chunk_number, n_heads, n_groups, layer_num, mamba_every_n_block, dropout)

        # CTC head
        self.ctc_conv = CTCConv(encoder_d_model, encoder_d_model, 7)
        self.ctc_classifier = nn.Sequential(nn.Linear(encoder_d_model, 2 * encoder_d_model),
                                            nn.SiLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(2 * encoder_d_model, num_vocab))

        # RNNT part
        self.rnnt_decoder = Decoder(num_vocab, predictor_d_model, predictor_d_model)
        self.rnnt_classifier = SimpleJoiner(encoder_d_model, predictor_d_model, joiner_d_model, num_vocab, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, audio_len, targets, targets_len):
        # [B, C, T, F]
        x = self.features_extractor(x)
        x = self.spec_aug(x, audio_len)
        x = self.downsample_conv(x)
        # x = self.dropout(x)
        enc_out = self.encoder(x, audio_len)
        # enc_out = self.dropout(enc_out)

        # CTC head
        ctc_logits = self.ctc_conv(enc_out)
        ctc_logits = self.dropout(ctc_logits)
        ctc_logits = self.ctc_classifier(ctc_logits)

        # RNNT head
        dec_out = self.rnnt_decoder(targets, targets_len)
        # dec_out = self.dropout(dec_out)
        rnnt_logits = self.rnnt_classifier(enc_out, dec_out)

        return ctc_logits, rnnt_logits
    
    def infer_ctc(self, x, audio_len=None, blank_token_id=1024):
        # [B, C, T, F]
        x = self.features_extractor(x)
        x = self.downsample_conv(x)
        x = self.encoder(x, audio_len)
        enc_out = self.post_norm(x)

        # CTC head
        ctc_logits = self.ctc_decoder.infer(enc_out)
        ctc_logits = self.post_ctc_dec_norm(ctc_logits)
        ctc_logits = self.ctc_classifier(ctc_logits)

        out = ctc_greedy_decode_batch(ctc_logits, blank_token_id)

        return out

    def infer_rnnt(self, x, audio_len=None, blank_token_id=1024):
        # [B, C, T, F]
        assert x.shape[0] == 1, "Now only single file supported"

        x = self.features_extractor(x)
        x = self.downsample_conv(x)
        x = self.encoder(x, audio_len)
        enc_out = self.post_norm(x)

        # RNNT greedy decoding
        dec_out, hidden = self._run_decoder(torch.tensor(blank_token_id))
        ans = []
        max_gen_tokens = 10  # Redundant in most cases
        for t in range(enc_out.shape[1]):
            enc_out_t = enc_out[:, t: t + 1, :]

            step = 0
            while step < max_gen_tokens:
                step += 1
                token = self._run_joiner(enc_out_t, dec_out)

                if token.item() != blank_token_id:
                    ans.append(token.item())
                    dec_out, hidden = self._run_decoder(token, hidden)

                else:
                    break

        return ans

    def _run_decoder(self, token, hidden=None):
        token = token.unsqueeze(0).unsqueeze(0)
        dec_out, hidden = self.rnnt_decoder.infer(token, hidden)
        dec_out = self.post_rnnt_dec_norm(dec_out)

        return dec_out, hidden

    def _run_joiner(self, enc_out, dec_out):
        out = self.rnnt_classifier(enc_out, dec_out)
        out = torch.argmax(torch.squeeze(out))
        return out


def count_parameters(model):
    parametrs_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parametrs_num} ({parametrs_num / (10 ** 6):.1f}M)")


if __name__ == '__main__':
    torch.manual_seed(21)

    time_factor = 4
    chunk_size = 100
    n_freq = 161
    model = ConformerCTC(num_vocab=1025,
                         inter_d_model=768,
                         n_mel=60,
                         time_factor=time_factor,
                         chunk_size=chunk_size,
                         freq_dim=n_freq,
                         n_heads=12,
                         layer_num=12,
                         dropout=0.1)

    count_parameters(model)

    max_len = 145
    dummy_input = torch.randn(2, 2, max_len, n_freq)
    res = model(dummy_input, torch.tensor([max_len, 93]))
    print(f"There are NANs: {torch.isnan(res).any()}")
    print(f"There are INFs: {torch.isinf(res).any()}")
    print(res.shape)