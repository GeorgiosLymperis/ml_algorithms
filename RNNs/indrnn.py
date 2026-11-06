import torch
import torch.nn as nn

class IndRNNCell(nn.Module):
    def __init__(self, input_dim, units, recurrent_clip=1.0):
        super().__init__()
        self.recurrent_clip = recurrent_clip
        self.w_xh = nn.Linear(input_dim, units)
        self.u = nn.Parameter(torch.empty(units).uniform_(0, recurrent_clip))

    def forward(self, x_t, h_prev):
        u_clipped = torch.clamp(self.u, -self.recurrent_clip, self.recurrent_clip)
        return torch.relu(self.w_xh(x_t) + u_clipped * h_prev)

    def clip_recurrent(self):
        with torch.no_grad():
            self.u.clamp_(-self.recurrent_clip, self.recurrent_clip)



class IndRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, units, num_layers):
        super().__init__()
        self.units = units
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.cells = nn.ModuleList([
            IndRNNCell(embed_dim if i == 0 else units, units) for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(units, vocab_size)
        self.norms = nn.ModuleList([nn.BatchNorm1d(units) for _ in range(num_layers)])

    def clip_recurrent(self):
        for cell in self.cells:
            cell.clip_recurrent()

    def forward(self, x):
        batch, seq_len = x.shape
        layer_input = self.embed(x)
        for cell, norm in zip(self.cells, self.norms):
            h = torch.zeros(batch, self.units, device=x.device)
            outputs = []
            for t in range(seq_len):
                h = cell(layer_input[:, t], h)
                outputs.append(h)
            out = torch.stack(outputs, dim=1)  # (B, T, H)
            layer_input = norm(out.flatten(0, 1)).view_as(out)

        return self.out_proj(layer_input)