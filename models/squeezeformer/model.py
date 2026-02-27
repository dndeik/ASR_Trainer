import torch
import torch.nn as nn

from models.squeezeformer.encoder import AudioEncoder
from models.squeezeformer.conv_module import CausalConv1d
from utils.decoding import ctc_greedy_decode_batch

# from encoder import AudioEncoder


class TransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1):
        super().__init__()

        self.conv = CausalConv1d(in_channels, out_channels, k_size, stride=stride, conv_style="full")

    def forward(self, x):
        """[B, T, C]"""
        x = x.transpose(1, 2)
        x = self.conv(x)  # [B, C, T//stride]
        x = x.transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, num_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, embed_dim)
        self.predictor = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=2,
                                 dropout=0.05)

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
        joint = self.dropout(joint)

        out = self.fc_out(joint)  # [B, T, U, num_vocab]
        return out


class ConformerHybrid(nn.Module):
    def __init__(self, num_vocab, encoder_d_model,
                 predictor_d_model, joiner_d_model, freq_dim, n_mel, time_factor, n_heads, n_groups, chunk_size,
                 left_context_chunk_number=1, right_context_chunk_number=0,
                 layer_num=8, mamba_every_n_block=3, dropout=0.1):
        super().__init__()
        # Encoder part
        self.time_factor = time_factor
        self.first_conv_kernel = self.time_factor * 2 + 1
        downsampled_chunk_size = chunk_size // self.time_factor

        self.downsample_conv = TransposedConv(n_mel, encoder_d_model, k_size=self.first_conv_kernel, stride=self.time_factor)
        self.encoder = AudioEncoder(encoder_d_model, downsampled_chunk_size, left_context_chunk_number,
                                    right_context_chunk_number, n_heads, n_groups, 9, layer_num, mamba_every_n_block,
                                    dropout)

        # CTC head
        self.ctc_conv = TransposedConv(encoder_d_model, encoder_d_model, 7)
        self.ctc_classifier = nn.Sequential(nn.Linear(encoder_d_model, 2 * encoder_d_model),
                                            nn.SiLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(2 * encoder_d_model, num_vocab))

        # RNNT part
        self.rnnt_decoder = Decoder(num_vocab, predictor_d_model, predictor_d_model)
        self.rnnt_classifier = SimpleJoiner(encoder_d_model, predictor_d_model, joiner_d_model, num_vocab,
                                            dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, audio_len, targets, targets_len):
        # [B, L, D]
        enc_out, corrected_len = self.run_encoder(x, audio_len)

        # CTC head
        ctc_logits = self.run_ctc_head(enc_out)

        # RNNT head
        dec_out = self.rnnt_decoder(targets, targets_len)
        rnnt_logits = self.rnnt_classifier(enc_out, dec_out)

        return ctc_logits, rnnt_logits, corrected_len

    def run_encoder(self, x, audio_len):
        # [B, L, D]
        x = self.downsample_conv(x)
        corrected_lens = self._calculate_correct_lens(audio_len, tensor_len=x.shape[1])
        enc_out = self.encoder(x, corrected_lens)
        return enc_out, corrected_lens

    def run_ctc_head(self, enc_out):
        ctc_logits = self.ctc_conv(enc_out)
        ctc_logits = self.dropout(ctc_logits)
        ctc_logits = self.ctc_classifier(ctc_logits)
        return ctc_logits

    def _calculate_correct_lens(self, audio_len, tensor_len):
        if audio_len is None:
            return audio_len
        corrected_lens = (audio_len + self.first_conv_kernel // 2) // self.time_factor + 1
        corrected_lens = torch.clamp(corrected_lens, max=tensor_len)
        return corrected_lens

    def infer_ctc(self, x, audio_len=None, blank_token_id=1024):
        # [B, C, T, F]
        enc_out, _ = self.run_encoder(x, audio_len)

        # CTC head
        ctc_logits = self.run_ctc_head(enc_out)

        out = ctc_greedy_decode_batch(ctc_logits, blank_token_id)

        return out

    def infer_rnnt(self, x, audio_len=None, blank_token_id=1024):
        # [B, C, T, F]
        assert x.shape[0] == 1, "Now only single file supported"

        enc_out, _ = self.run_encoder(x, audio_len)

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
    freq_dim = 161
    model = ConformerHybrid(num_vocab=1025,
                            encoder_d_model=768,
                            predictor_d_model=768,
                            joiner_d_model=768,
                            freq_dim=freq_dim,
                            n_mel=60,
                            time_factor=time_factor,
                            chunk_size=chunk_size,
                            n_heads=12,
                            n_groups=6,
                            layer_num=12,
                            dropout=0.1)

    count_parameters(model)

    max_len = 1500
    dummy_input = torch.randn(2, 2, max_len, freq_dim)
    lens = torch.tensor([max_len, 1200])
    res = model.run_encoder(dummy_input, lens // time_factor)
    print(f"There are NANs: {torch.isnan(res).any()}")
    print(f"There are INFs: {torch.isinf(res).any()}")
    print(res.shape)