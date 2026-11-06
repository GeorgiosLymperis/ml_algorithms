import torch
import torch.nn as nn
from lstm import LSTMCell

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, units):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = LSTMCell(embed_dim, units)

    def forward(self, x, lengths=None):
        batch, seq_len = x.shape
        if lengths is None:
            lengths = torch.full((batch,), seq_len)
        lengths = lengths.to(x.device)
        emb = self.embed(x)  # (B, T, E)
        h = torch.zeros(batch, self.units, device=x.device)
        c = torch.zeros(batch, self.units, device=x.device)
        for t in range(seq_len):
            valid = (t < lengths).unsqueeze(-1)  # (B, 1)
            h_new, c_new = self.cell(emb[:, t], h, c)
            h = torch.where(valid, h_new, h)
            c = torch.where(valid, c_new, c)
        return h, c  # (B, H)



class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, units):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cell = LSTMCell(embed_dim, units)
        self.out_proj = nn.Linear(units, vocab_size)

    def forward(self, x, h, c):
        batch, seq_len = x.shape
        emb = self.embed(x)
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(emb[:, t], h, c)
            outputs.append(h)
        return self.out_proj(torch.stack(outputs, dim=1)), h, c


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, units):
        super().__init__()
        self.encoder = Seq2SeqEncoder(src_vocab_size, embed_dim, units)
        self.decoder = Seq2SeqDecoder(tgt_vocab_size, embed_dim, units)

    def forward(self, src, dec_in, src_lengths=None):
        h, c = self.encoder(src, src_lengths)
        logits, _, _ = self.decoder(dec_in, h, c)
        return logits

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len, src_lengths=None):
        h, c = self.encoder(src, src_lengths)
        token = torch.full((src.size(0), 1), bos_id, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        tokens = []
        for _ in range(max_len):
            logits, h, c = self.decoder(token, h, c)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
            finished |= token.squeeze(1) == eos_id
            if finished.all():
                break
        return torch.cat(tokens, dim=1)  # (B, <=max_len)
