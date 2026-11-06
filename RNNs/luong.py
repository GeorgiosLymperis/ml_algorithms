import torch
import torch.nn as nn
from lstm import LSTMCell

class LuongAttention(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.w_a = nn.Linear(units, units, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        query = self.w_a(dec_hidden).unsqueeze(1)  # (B, 1, H)
        scores = (query * enc_outputs).sum(dim=-1)  # (B, S)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, S)
        context = (weights.unsqueeze(-1) * enc_outputs).sum(dim=1)  # (B, H)
        return context, weights


class LuongCell(nn.Module):
    def __init__(self, embed_dim, units):
        super().__init__()
        self.cell = LSTMCell(embed_dim + units, units)
        self.attention = LuongAttention(units)
        self.combine = nn.Linear(2 * units, units)

    def forward(self, x_t, h_prev, c_prev, h_tilde_prev, enc_outputs, mask=None):
        step_input = torch.cat([x_t, h_tilde_prev], dim=-1)
        h, c = self.cell(step_input, h_prev, c_prev)
        context, weights = self.attention(h, enc_outputs, mask)
        h_tilde = torch.tanh(self.combine(torch.cat([h, context], dim=-1)))
        return h_tilde, h, c, weights


class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, units):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.luong_cell = LuongCell(embed_dim, units)
        self.out_proj = nn.Linear(units, vocab_size)

    def forward(self, dec_inputs, enc_outputs, h, c, h_tilde=None, mask=None):
        if h_tilde is None:
            h_tilde = torch.zeros_like(h)
        emb = self.embed(dec_inputs)
        outputs, all_weights = [], []
        for t in range(dec_inputs.size(1)):
            h_tilde, h, c, weights = self.luong_cell(emb[:, t], h, c, h_tilde, enc_outputs, mask)
            outputs.append(h_tilde)
            all_weights.append(weights)
        logits = self.out_proj(torch.stack(outputs, dim=1))
        return logits, h, c, h_tilde, torch.stack(all_weights, dim=1)  # weights: (B, T, S)


class LuongAttentionModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, units):
        super().__init__()
        self.units = units
        self.enc_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.enc_cell = LSTMCell(embed_dim, units)
        self.decoder = LuongDecoder(tgt_vocab_size, embed_dim, units)

    def encode(self, src, src_lengths=None):
        batch, seq_len = src.shape
        if src_lengths is None:
            src_lengths = torch.full((batch,), seq_len)
        src_lengths = src_lengths.to(src.device)
        mask = torch.arange(seq_len, device=src.device).expand(batch, seq_len) < src_lengths.unsqueeze(1)
        emb = self.enc_embed(src)
        h = torch.zeros(batch, self.units, device=src.device)
        c = torch.zeros(batch, self.units, device=src.device)
        enc_outputs = []
        for t in range(seq_len):
            valid = mask[:, t].unsqueeze(-1)  # (B, 1)
            h_new, c_new = self.enc_cell(emb[:, t], h, c)
            h = torch.where(valid, h_new, h)
            c = torch.where(valid, c_new, c)
            enc_outputs.append(h * valid)
        return torch.stack(enc_outputs, dim=1), h, c, mask

    def forward(self, src, dec_in, src_lengths=None):
        enc_outputs, h, c, mask = self.encode(src, src_lengths)
        logits, _, _, _, weights = self.decoder(dec_in, enc_outputs, h, c, mask=mask)
        return logits, weights

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len, src_lengths=None):
        enc_outputs, h, c, mask = self.encode(src, src_lengths)
        h_tilde = None
        token = torch.full((src.size(0), 1), bos_id, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        tokens, all_weights = [], []
        for _ in range(max_len):
            logits, h, c, h_tilde, weights = self.decoder(token, enc_outputs, h, c, h_tilde, mask)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
            all_weights.append(weights)
            finished |= token.squeeze(1) == eos_id
            if finished.all():
                break
        return torch.cat(tokens, dim=1), torch.cat(all_weights, dim=1)  # (B, <=max_len), (B, <=max_len, S)
