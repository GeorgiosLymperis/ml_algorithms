import torch
import torch.nn as nn
from lstm import LSTMCell

def reverse_valid(x, lengths):
    """Reverses the valid steps of (B, T) or (B, T, H), leaving right padding in place."""
    batch, seq_len = x.shape[:2]
    steps = torch.arange(seq_len, device=x.device).expand(batch, seq_len)
    lengths = lengths.to(x.device).unsqueeze(1)
    rev = torch.where(steps < lengths, lengths - 1 - steps, steps)
    if x.dim() == 3:
        rev = rev.unsqueeze(-1).expand_as(x)
    return x.gather(1, rev)


class ELMo(nn.Module):
    def __init__(self, vocab_size, embed_dim, units):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.token_proj = nn.Linear(embed_dim, units)
        self.fwd_cell1 = LSTMCell(units, units)
        self.fwd_cell2 = LSTMCell(units, units)
        self.bwd_cell1 = LSTMCell(units, units)
        self.bwd_cell2 = LSTMCell(units, units)
        self.fwd_out = nn.Linear(units, vocab_size)
        self.bwd_out = nn.Linear(units, vocab_size)

    def _run_stack(self, cell1, cell2, x):
        batch, seq_len, _ = x.shape
        h1 = torch.zeros(batch, self.units, device=x.device)
        c1 = torch.zeros(batch, self.units, device=x.device)
        layer1_outputs = []
        for t in range(seq_len):
            h1, c1 = cell1(x[:, t], h1, c1)
            layer1_outputs.append(h1)
        layer1 = torch.stack(layer1_outputs, dim=1)

        h2 = torch.zeros(batch, self.units, device=x.device)
        c2 = torch.zeros(batch, self.units, device=x.device)
        layer2_outputs = []
        for t in range(seq_len):
            h2, c2 = cell2(layer1[:, t], h2, c2)
            layer2_outputs.append(h2)
        layer2 = torch.stack(layer2_outputs, dim=1) + layer1  # residual connection
        return layer1, layer2

    def forward(self, fwd_input, bwd_input):
        _, fwd2 = self._run_stack(self.fwd_cell1, self.fwd_cell2, self.token_proj(self.embed(fwd_input)))
        _, bwd2 = self._run_stack(self.bwd_cell1, self.bwd_cell2, self.token_proj(self.embed(bwd_input)))
        return self.fwd_out(fwd2), self.bwd_out(bwd2)

    def layer_outputs(self, tokens, lengths=None):
        """Returns the three ELMo layers, each (B, T, 2 * units) and aligned with tokens."""
        batch, seq_len = tokens.shape
        if lengths is None:
            lengths = torch.full((batch,), seq_len)
        lengths = lengths.to(tokens.device)
        x = self.token_proj(self.embed(tokens))  # (B, T, H)
        fwd1, fwd2 = self._run_stack(self.fwd_cell1, self.fwd_cell2, x)
        bwd1, bwd2 = self._run_stack(self.bwd_cell1, self.bwd_cell2, reverse_valid(x, lengths))
        bwd1, bwd2 = reverse_valid(bwd1, lengths), reverse_valid(bwd2, lengths)
        mask = torch.arange(seq_len, device=tokens.device).expand(batch, seq_len) < lengths.unsqueeze(1)
        layers = [
            torch.cat([x, x], dim=-1),
            torch.cat([fwd1, bwd1], dim=-1),
            torch.cat([fwd2, bwd2], dim=-1),
        ]
        return [layer * mask.unsqueeze(-1) for layer in layers]


class ELMoEmbedder(nn.Module):
    """ELMo_k = gamma * sum_j(s_j * h_{k,j}), with the pretrained biLM frozen."""

    def __init__(self, elmo):
        super().__init__()
        self.elmo = elmo
        self.elmo.requires_grad_(False)
        self.scalars = nn.Parameter(torch.zeros(3))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, tokens, lengths=None):
        with torch.no_grad():
            layers = self.elmo.layer_outputs(tokens, lengths)
        s = torch.softmax(self.scalars, dim=0)
        return self.gamma * sum(s[j] * layers[j] for j in range(3))  # (B, T, 2 * units)
