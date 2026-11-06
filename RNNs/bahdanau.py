import torch
import torch.nn as nn
from gru import GRUCell, BiGRU

class BahdanauAttention(nn.Module):
    """
    score(h_dec, h_enc_j) = v^T tanh(Wa h_enc_j + Ua h_dec)   # additive/"Bahdanau" scoring
    alpha_j = softmax_j(score(h_dec, h_enc_j))                # attention weights over source positions
    context = sum_j(alpha_j * h_enc_j)                        # weighted sum = context vector
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.U_a = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        dec_hidden_expanded = dec_hidden.unsqueeze(1)  # (B, 1, D)
        scores = self.v_a(torch.tanh(self.W_a(enc_outputs) + self.U_a(dec_hidden_expanded)))  # (B, S, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, S, 1)
        context = (weights * enc_outputs).sum(dim=1)  # (B, E)
        return context, weights.squeeze(-1)


class AttentionCell(nn.Module):
    def __init__(self, embed_dim, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.cell = GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

    def forward(self, x_t, h_prev, enc_outputs, mask=None):
        context, weights = self.attention(h_prev, enc_outputs, mask)
        step_input = torch.cat([x_t, context], dim=-1)
        return self.cell(step_input, h_prev), weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn_cell = AttentionCell(embed_dim, encoder_dim, decoder_dim, attention_dim)
        self.out_proj = nn.Linear(decoder_dim, vocab_size)

    def forward(self, dec_inputs, enc_outputs, h, mask=None):
        emb = self.embed(dec_inputs)
        outputs, all_weights = [], []
        for t in range(dec_inputs.size(1)):
            h, weights = self.attn_cell(emb[:, t], h, enc_outputs, mask)
            outputs.append(h)
            all_weights.append(weights)
        logits = self.out_proj(torch.stack(outputs, dim=1))
        return logits, h, torch.stack(all_weights, dim=1)  # weights: (B, T, S)


class Seq2SeqBahdanau(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder = BiGRU(embed_dim, encoder_dim // 2)  # outputs are encoder_dim wide
        self.bridge = nn.Linear(encoder_dim // 2, decoder_dim)
        self.decoder = AttentionDecoder(tgt_vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim)

    def encode(self, src, src_lengths=None):
        batch, seq_len = src.shape
        if src_lengths is None:
            src_lengths = torch.full((batch,), seq_len)
        src_lengths = src_lengths.to(src.device)
        mask = torch.arange(seq_len, device=src.device).expand(batch, seq_len) < src_lengths.unsqueeze(1)
        enc_outputs, h_n = self.encoder(self.enc_embed(src), src_lengths)
        h = torch.tanh(self.bridge(h_n[1]))  # start from the backward final state
        return enc_outputs, h, mask

    def forward(self, src, dec_in, src_lengths=None):
        enc_outputs, h, mask = self.encode(src, src_lengths)
        logits, _, weights = self.decoder(dec_in, enc_outputs, h, mask)
        return logits, weights

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len, src_lengths=None):
        enc_outputs, h, mask = self.encode(src, src_lengths)
        token = torch.full((src.size(0), 1), bos_id, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        tokens, all_weights = [], []
        for _ in range(max_len):
            logits, h, weights = self.decoder(token, enc_outputs, h, mask)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
            all_weights.append(weights)
            finished |= token.squeeze(1) == eos_id
            if finished.all():
                break
        return torch.cat(tokens, dim=1), torch.cat(all_weights, dim=1)  # (B, <=max_len), (B, <=max_len, S)