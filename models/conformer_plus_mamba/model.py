import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.conformer_plus_mamba.encoder import AudioEncoder

# from encoder import AudioEncoder

GLOBAL_EPS = 1e-3


class SpeechEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, stride=1):
        super().__init__()

        self.t_pad = k_size - stride
        self.conv_padding = nn.ConstantPad1d((self.t_pad, 0), 0)

        self.conv = nn.Sequential(nn.Conv1d(in_channel, out_channel, k_size, stride=stride),
                                  nn.BatchNorm1d(out_channel, eps=GLOBAL_EPS))

    def forward(self, x):
        """[B, T, C]"""
        x = x.transpose(1, 2)
        x = self.conv_padding(x)
        x = self.conv(x)  # [B, C, T//stride]
        x = x.transpose(1, 2)
        return x


class ConformerCTC(nn.Module):
    def __init__(self, num_vocab, inter_d_model, n_mel, time_factor, chunk_size, freq_dim, n_heads=4, layer_num=8,
                 dropout=0.1):
        super().__init__()
        chunk_size = chunk_size // time_factor
        self.trainable_mel = self._get_trainable_mel(freq_dim, n_mel)
        self.in_proj = SpeechEncoder(n_mel, inter_d_model, k_size=time_factor*2+1, stride=time_factor)
        self.encoder = AudioEncoder(inter_d_model, chunk_size, 1, n_heads, layer_num, dropout)

        self.post_norm = nn.LayerNorm(inter_d_model, eps=GLOBAL_EPS)
        self.ctc_classifier = nn.Sequential(nn.Linear(inter_d_model, 4 * inter_d_model),
                                            nn.SiLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(4 * inter_d_model, num_vocab))

    def _get_trainable_mel(self, freq_dim, n_mel):
        mel_fb = torchaudio.functional.melscale_fbanks(n_freqs=freq_dim, f_min=0, f_max=8000, n_mels=n_mel,
                                                       sample_rate=16000)
        linear = nn.Linear(freq_dim, n_mel, bias=False)
        linear.weight.data = mel_fb.T.data
        return linear

    def forward(self, x, audio_len):
        # [B, C, T, F]
        x = torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)  # mag [B, T, F]
        x = self.trainable_mel(x)
        # x = torch.log(x+GLOBAL_EPS)
        x = self.in_proj(x)
        x = self.encoder(x, audio_len)
        x = self.post_norm(x)
        x = self.ctc_classifier(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_vocab, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, embed_dim)
        #self.predictor = nn.GRU(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=2, dropout=dropout)
        self.predictor = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=1)

    def forward(self, y, y_lengths):
        emb = self.embedding(y)
        y_lengths = y_lengths.cpu()
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb, y_lengths, batch_first=True, enforce_sorted=False)
        packed_seq, _ = self.predictor(packed_seq)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
        return out

class CTCDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.predictor = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lengths):
        x_lengths = x_lengths.cpu()
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        packed_seq, _ = self.predictor(packed_seq)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
        out = self.dropout(out)
        return out


class JointNet(nn.Module):
    def __init__(self, enc_dim, dec_dim, joint_dim, num_vocab):
        super().__init__()
        self.fc_enc = nn.Linear(enc_dim, joint_dim)
        self.fc_pred = nn.Linear(dec_dim, joint_dim)
        self.fc_out = nn.Linear(joint_dim * 2, num_vocab)

    def forward(self, enc_out, pred_out):
        enc_out = self.fc_enc(enc_out).unsqueeze(2)
        pred_out = self.fc_pred(pred_out).unsqueeze(1)

        T = enc_out.size(1)
        U = pred_out.size(2)

        enc_exp = enc_out.expand(-1, T, U, -1)  # [B, T, U, joint_dim]
        pred_exp = pred_out.expand(-1, T, U, -1)  # [B, T, U, joint_dim]

        joint = torch.cat([enc_exp, pred_exp], dim=-1)  # [B, T, U, joint_dim * 2]

        out = self.fc_out(joint)  # [B, T, U, num_vocab]
        return out


class ConformerHybrid(nn.Module):
    def __init__(self, num_vocab, inter_d_model, n_mel, time_factor, chunk_size, freq_dim, n_heads=4, layer_num=8,
                 dropout=0.1):
        super().__init__()
        # Encoder part
        chunk_size = chunk_size // time_factor
        self.trainable_mel = self._get_trainable_mel(freq_dim, n_mel)
        self.in_proj = SpeechEncoder(n_mel, inter_d_model, k_size=time_factor*2+1, stride=time_factor)
        self.encoder = AudioEncoder(inter_d_model, chunk_size, 1, n_heads, layer_num, dropout)
        self.post_norm = nn.LayerNorm(inter_d_model, eps=GLOBAL_EPS)

        # CTC head
        self.ctc_decoder = CTCDecoder(inter_d_model, inter_d_model, dropout=dropout)
        self.post_ctc_dec_norm = nn.LayerNorm(inter_d_model, eps=GLOBAL_EPS)
        self.ctc_classifier = nn.Sequential(nn.Linear(inter_d_model, 2 * inter_d_model),
                                            nn.SiLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(2 * inter_d_model, num_vocab))

        # RNNT part
        self.rnnt_decoder = Decoder(num_vocab, inter_d_model, inter_d_model)
        self.post_rnnt_dec_norm = nn.LayerNorm(inter_d_model, eps=GLOBAL_EPS)
        self.rnnt_classifier = JointNet(inter_d_model, inter_d_model, int(1.5 * inter_d_model), num_vocab)

    def _get_trainable_mel(self, freq_dim, n_mel):
        mel_fb = torchaudio.functional.melscale_fbanks(n_freqs=freq_dim, f_min=0, f_max=8000, n_mels=n_mel,
                                                       sample_rate=16000)
        linear = nn.Linear(freq_dim, n_mel, bias=False)
        linear.weight.data = mel_fb.T.data
        return linear

    def forward(self, x, audio_len, targets, targets_len):
        # [B, C, T, F]
        x = torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)  # mag [B, T, F]
        x = self.trainable_mel(x)
        x = self.in_proj(x)
        x = self.encoder(x, audio_len)
        enc_out = self.post_norm(x)

        # CTC head
        ctc_logits = self.ctc_decoder(enc_out, audio_len)
        ctc_logits = self.post_ctc_dec_norm(ctc_logits)
        ctc_logits = self.ctc_classifier(ctc_logits)

        # RNNT head
        dec_out = self.rnnt_decoder(targets, targets_len)
        dec_out = self.post_rnnt_dec_norm(dec_out)
        rnnt_logits = self.rnnt_classifier(enc_out, dec_out)

        return ctc_logits, rnnt_logits


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