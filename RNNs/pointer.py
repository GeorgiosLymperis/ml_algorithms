import torch
import torch.nn as nn
from lstm import LSTMCell

class PointerAttention(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.w1 = nn.Linear(units, units, bias=False)
        self.w2 = nn.Linear(units, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        dec_hidden_expanded = dec_hidden.unsqueeze(1)  # (B, 1, H)
        scores = self.v(torch.tanh(self.w1(enc_outputs) + self.w2(dec_hidden_expanded))).squeeze(-1)  # (B, S)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores  # logits over input positions


class PointerCell(nn.Module):
    def __init__(self, embed_dim, units):
        super().__init__()
        self.cell = LSTMCell(embed_dim, units)
        self.pointer_attn = PointerAttention(units)

    def forward(self, x_t, h_prev, c_prev, enc_outputs, mask=None):
        h, c = self.cell(x_t, h_prev, c_prev)
        logits = self.pointer_attn(h, enc_outputs, mask)
        return logits, h, c


class ValueEncoder(nn.Module):
    def __init__(self, embed_dim, units):
        super().__init__()
        self.units = units
        self.value_proj = nn.Linear(1, embed_dim)
        self.cell = LSTMCell(embed_dim, units)

    def forward(self, values, mask):
        batch, seq_len = values.shape
        x = self.value_proj(values.unsqueeze(-1))
        h = torch.zeros(batch, self.units, device=values.device)
        c = torch.zeros(batch, self.units, device=values.device)
        outputs = []
        for t in range(seq_len):
            valid = mask[:, t].unsqueeze(-1)  # (B, 1)
            h_new, c_new = self.cell(x[:, t], h, c)
            h = torch.where(valid, h_new, h)
            c = torch.where(valid, c_new, c)
            outputs.append(h * valid)
        return torch.stack(outputs, dim=1), h, c


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim, units):
        super().__init__()
        self.units = units
        self.value_proj = nn.Linear(1, embed_dim)
        self.ptr_cell = PointerCell(embed_dim, units)

    def forward(self, dec_values, enc_outputs, h, c, mask=None):
        x = self.value_proj(dec_values.unsqueeze(-1))
        logits_seq = []
        for t in range(dec_values.size(1)):
            logits, h, c = self.ptr_cell(x[:, t], h, c, enc_outputs, mask)
            logits_seq.append(logits)
        return torch.stack(logits_seq, dim=1), h, c  # (B, T, S)


class PointerNetwork(nn.Module):
    def __init__(self, embed_dim, units):
        super().__init__()
        self.value_encoder = ValueEncoder(embed_dim, units)
        self.decoder = PointerDecoder(embed_dim, units)

    def encode(self, enc_values, lengths=None):
        batch, seq_len = enc_values.shape
        if lengths is None:
            lengths = torch.full((batch,), seq_len)
        lengths = lengths.to(enc_values.device)
        mask = torch.arange(seq_len, device=enc_values.device).expand(batch, seq_len) < lengths.unsqueeze(1)
        enc_outputs, h, c = self.value_encoder(enc_values, mask)
        return enc_outputs, h, c, mask, lengths

    def forward(self, enc_values, dec_values, lengths=None):
        enc_outputs, h, c, mask, _ = self.encode(enc_values, lengths)
        logits, _, _ = self.decoder(dec_values, enc_outputs, h, c, mask)
        return logits

    @torch.no_grad()
    def generate(self, enc_values, start_value, lengths=None, block_visited=True):
        enc_outputs, h, c, mask, lengths = self.encode(enc_values, lengths)
        value = torch.full((enc_values.size(0), 1), start_value, device=enc_values.device)
        visited = torch.zeros_like(mask)
        indices = []
        for t in range(lengths.max().item()):
            step_mask = mask & ~visited if block_visited else mask
            logits, h, c = self.decoder(value, enc_outputs, h, c, step_mask)
            idx = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
            visited = visited.scatter(1, idx, True)
            value = enc_values.gather(1, idx)
            indices.append(torch.where(t < lengths.unsqueeze(1), idx, -1))
        return torch.cat(indices, dim=1)  # (B, max(lengths)), -1 after a sequence ends
