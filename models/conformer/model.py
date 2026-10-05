import torch
import torch.nn as nn

from models.conformer.encoder import AudioEncoder
from models.conformer.conv_module import CausalConv1d
from models.common_modules.transformer_predictor import TransformerDecoder
from utils.decoding import ctc_greedy_decode_batch


class TransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, k_size, stride=stride, conv_style="full")

    def forward(self, x):
        """Apply a causal convolution to a [B, T, C] tensor."""
        x = self.conv(x.transpose(1, 2))
        return x.transpose(1, 2)


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
        joint = self.act(enc_out + pred_out)
        # Match the squeezeformer joiner regularization.
        return self.fc_out(self.dropout(joint))


class ConformerHybrid(nn.Module):
    def __init__(self, num_vocab, encoder_d_model, predictor_d_model, joiner_d_model,
                 n_mel, time_factor, n_heads, n_groups, chunk_size,
                 left_context_chunk_number=1, right_context_chunk_number=0,
                 encoder_layer_num=8, predictor_layer_num=1, mamba_every_n_block=3,
                 dropout=0.1, drop_path=0.1):
        super().__init__()
        self.time_factor = time_factor
        self.first_conv_kernel = time_factor * 2 + 1
        downsampled_chunk_size = chunk_size // time_factor

        self.downsample_conv = TransposedConv(
            n_mel, encoder_d_model, k_size=self.first_conv_kernel, stride=time_factor
        )
        self.encoder = AudioEncoder(
            encoder_d_model, downsampled_chunk_size, left_context_chunk_number,
            right_context_chunk_number, n_heads, n_groups, encoder_layer_num,
            mamba_every_n_block, dropout,
        )

        self.ctc_conv = TransposedConv(encoder_d_model, encoder_d_model, 7)
        self.ctc_classifier = nn.Sequential(
            nn.Linear(encoder_d_model, 2 * encoder_d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * encoder_d_model, num_vocab),
        )

        self.rnnt_decoder = TransformerDecoder(
            num_vocab,
            predictor_d_model,
            predictor_d_model,
            layer_num=predictor_layer_num,
            dropout=dropout,
            num_heads=8,
            num_groups=8,
            context_size=32,
        )
        self.rnnt_classifier = SimpleJoiner(
            encoder_d_model, predictor_d_model, joiner_d_model, num_vocab,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, audio_len, targets, targets_len):
        enc_out, corrected_len = self.run_encoder(x, audio_len)
        ctc_logits = self.run_ctc_head(enc_out)
        dec_out = self.rnnt_decoder(targets, targets_len)
        rnnt_logits = self.rnnt_classifier(enc_out, dec_out)
        return ctc_logits, rnnt_logits, corrected_len

    def run_encoder(self, x, audio_len):
        x = self.downsample_conv(x)
        corrected_lens = self._calculate_correct_lens(audio_len, x.shape[1])
        return self.encoder(x, corrected_lens), corrected_lens

    def run_ctc_head(self, enc_out):
        return self.ctc_classifier(self.dropout(self.ctc_conv(enc_out)))

    def _calculate_correct_lens(self, audio_len, tensor_len):
        if audio_len is None:
            return None
        corrected_lens = (audio_len + self.first_conv_kernel // 2) // self.time_factor + 1
        return torch.clamp(corrected_lens, max=tensor_len)

    def infer_ctc(self, x, audio_len=None, blank_token_id=1024):
        enc_out, _ = self.run_encoder(x, audio_len)
        return ctc_greedy_decode_batch(self.run_ctc_head(enc_out), blank_token_id)

    def infer_rnnt(self, x, audio_len=None, blank_token_id=1024):
        assert x.shape[0] == 1, "Now only single file supported"
        enc_out, _ = self.run_encoder(x, audio_len)
        dec_out, hidden = self._run_decoder(torch.tensor(blank_token_id, device=x.device))
        ans = []
        for t in range(enc_out.shape[1]):
            enc_out_t = enc_out[:, t:t + 1, :]
            for _ in range(10):
                token = self._run_joiner(enc_out_t, dec_out)
                if token.item() == blank_token_id:
                    break
                ans.append(token.item())
                dec_out, hidden = self._run_decoder(token, hidden)
        return ans

    def _run_decoder(self, token, hidden=None):
        token = token.unsqueeze(0).unsqueeze(0)
        return self.rnnt_decoder.infer(token, hidden)

    def _run_joiner(self, enc_out, dec_out):
        return torch.argmax(torch.squeeze(self.rnnt_classifier(enc_out, dec_out)))


def count_parameters(model):
    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parameters_num} ({parameters_num / 10**6:.1f}M)")


if __name__ == '__main__':
    torch.manual_seed(21)
    model = ConformerHybrid(
        num_vocab=32,
        encoder_d_model=32,
        predictor_d_model=32,
        joiner_d_model=32,
        n_mel=80,
        time_factor=4,
        n_heads=4,
        n_groups=2,
        chunk_size=16,
        left_context_chunk_number=1,
        encoder_layer_num=2,
        predictor_layer_num=2,
        mamba_every_n_block=-1,
        dropout=0.0,
        drop_path=0.0,
    ).eval()
    count_parameters(model)

    features = torch.randn(2, 64, 80)
    audio_lengths = torch.tensor([64, 48])
    targets = torch.randint(0, 31, (2, 4))
    target_lengths = torch.tensor([4, 3])
    with torch.no_grad():
        ctc_logits, rnnt_logits, encoded_lengths = model(
            features, audio_lengths, targets, target_lengths
        )

    print(f"CTC logits: {tuple(ctc_logits.shape)}")
    print(f"RNNT logits: {tuple(rnnt_logits.shape)}")
    print(f"Encoded lengths: {encoded_lengths.tolist()}")
