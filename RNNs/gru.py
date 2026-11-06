import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """
    z_t = sigmoid(W_z [x_t, h_{t-1}])                    # update gate
    r_t = sigmoid(W_r [x_t, h_{t-1}])                    # reset gate
    h~_t = tanh(W_x x_t + r_t * (W_h h_{t-1}))            # candidate hidden state
    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t                # interpolated update
    """

    def __init__(self, input_dim, units):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.w_z = nn.Linear(input_dim + units, units)
        self.w_r = nn.Linear(input_dim + units, units)
        self.w_x = nn.Linear(input_dim, units)
        self.w_h = nn.Linear(units, units)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for gate in (self.w_z, self.w_r):
                nn.init.xavier_uniform_(gate.weight[:, :self.input_dim])
                nn.init.orthogonal_(gate.weight[:, self.input_dim:])
                nn.init.zeros_(gate.bias)
            nn.init.xavier_uniform_(self.w_x.weight)
            nn.init.zeros_(self.w_x.bias)
            nn.init.orthogonal_(self.w_h.weight)
            nn.init.zeros_(self.w_h.bias)

    def forward(self, x_t, h_prev):
        x_h = torch.cat([x_t, h_prev], dim=-1)
        z_t = torch.sigmoid(self.w_z(x_h))
        r_t = torch.sigmoid(self.w_r(x_h))
        h_candidate = torch.tanh(self.w_x(x_t) + r_t * self.w_h(h_prev))
        h_t = (1- z_t) * h_prev + z_t * h_candidate

        return h_t

    def count_parameters(self):
        return (
            2 * self.units * (self.input_dim + self.units + 1) +
            self.units * (self.input_dim + 1) +
            self.units * (self.units + 1)
        )


class BiGRU(nn.Module):
    """Returns per-step outputs (B, T, 2 * units) and final states (2, B, units)."""

    def __init__(self, input_dim, units, cell=GRUCell):
        super().__init__()
        self.units = units
        self.fwd = cell(input_dim, units)
        self.bwd = cell(input_dim, units)

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is None:
            lengths = torch.full((B,), T)
        lengths = lengths.to(x.device)
        mask = torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)  # (B, T)

        h_fwd = torch.zeros(B, self.units, device=x.device)  # (B, H)
        fwd_outputs = []

        for t in range(T):
            valid = mask[:, t].unsqueeze(-1)  # (B, 1)
            h_new = self.fwd(x[:, t, :], h_fwd)
            h_fwd = torch.where(valid, h_new, h_fwd)
            fwd_outputs.append(h_fwd * valid)

        h_bwd = torch.zeros(B, self.units, device=x.device)
        bwd_outputs = [None] * T
        for t in reversed(range(T)):
            valid = mask[:, t].unsqueeze(-1)
            h_new = self.bwd(x[:, t, :], h_bwd)
            h_bwd = torch.where(valid, h_new, h_bwd)
            bwd_outputs[t] = h_bwd * valid

        fwd_outputs = torch.stack(fwd_outputs, dim=1)  # (B, T, H)
        bwd_outputs = torch.stack(bwd_outputs, dim=1)  # (B, T, H)
        outputs = torch.cat([fwd_outputs, bwd_outputs], dim=-1)  # (B, T, 2H)
        final_states = torch.stack((h_fwd, h_bwd), dim=0)  # (2, B, H)
        return outputs, final_states

    def count_parameters(self):
        """Count total learnable parameters."""
        return self.fwd.count_parameters() + self.bwd.count_parameters()


class DeepBiGRU(nn.Module):
    """Returns per-step outputs (B, T, 2 * units) and final states (2 * num_layers, B, units)."""

    def __init__(self, input_dim, units, num_layers, cell=GRUCell):
        super().__init__()
        self.layers = nn.ModuleList([
            BiGRU(input_dim if i == 0 else 2 * units, units, cell)
            for i in range(num_layers)
        ])

    def forward(self, x, lengths=None):
        final_states = []
        for layer in self.layers:
            x, h_n = layer(x, lengths)  # x: (B, T, 2H), h_n: (2, B, H)
            final_states.append(h_n)
        return x, torch.cat(final_states, dim=0)  # (2 * num_layers, B, H)

    def count_parameters(self):
        """Count total learnable parameters."""
        return sum(layer.count_parameters() for layer in self.layers)


class DeepGRU(nn.Module):
    """Returns per-step outputs (B, T, units) and final states (num_layers, B, units)."""

    def __init__(self, input_dim, units, num_layers, cell=GRUCell):
        super().__init__()
        self.units = units
        self.cells = nn.ModuleList([
            cell(input_dim if i == 0 else units, units)
            for i in range(num_layers)
        ])

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is None:
            lengths = torch.full((B,), T)
        lengths = lengths.to(x.device)
        mask = torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)  # (B, T)

        final_states = []
        for cell in self.cells:
            h = torch.zeros(B, self.units, device=x.device)  # (B, H)
            outputs = []

            for t in range(T):
                valid = mask[:, t].unsqueeze(-1)  # (B, 1)
                h_new = cell(x[:, t, :], h)
                h = torch.where(valid, h_new, h)
                outputs.append(h * valid)

            x = torch.stack(outputs, dim=1)  # (B, T, H)
            final_states.append(h)

        return x, torch.stack(final_states, dim=0)  # (num_layers, B, H)

    def count_parameters(self):
        """Count total learnable parameters."""
        return sum(cell.count_parameters() for cell in self.cells)