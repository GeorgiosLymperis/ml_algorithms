import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """
    - i_t = sigmoid(W_i [x_t, h_{t-1}])      # input gate
    - f_t = sigmoid(W_f [x_t, h_{t-1}])      # forget gate
    - o_t = sigmoid(W_o [x_t, h_{t-1}])      # output gate
    - g_t = tanh(W_g [x_t, h_{t-1}])         # candidate value
    - c_t = f_t * c_{t-1} + i_t * g_t        # additive cell-state update
    - h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim, units):
        super().__init__()
        self.input_dim = input_dim
        self.units = units

        self.w_i = nn.Linear(input_dim + units, units)
        self.w_f = nn.Linear(input_dim + units, units)
        self.w_o = nn.Linear(input_dim + units, units)
        self.w_g = nn.Linear(input_dim + units, units)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for gate in (self.w_i, self.w_f, self.w_o, self.w_g):
                nn.init.xavier_uniform_(gate.weight[:, :self.input_dim])
                nn.init.orthogonal_(gate.weight[:, self.input_dim:])
                nn.init.zeros_(gate.bias)
            nn.init.ones_(self.w_f.bias)


    def forward(self, x_t, h_prev, c_prev):
        z = torch.cat([x_t, h_prev], dim=-1)
        i_t = torch.sigmoid(self.w_i(z))
        f_t = torch.sigmoid(self.w_f(z))
        o_t = torch.sigmoid(self.w_o(z))
        g_t = torch.tanh(self.w_g(z))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def count_parameters(self):
        """Count total learnable parameters."""
        return 4 * self.units * (self.input_dim + self.units + 1)

class PeepholeLSTMCell(nn.Module):
    """
    i_t = sigmoid(W_i [x_t, h_{t-1}] + p_i * c_{t-1})
    f_t = sigmoid(W_f [x_t, h_{t-1}] + p_f * c_{t-1})
    c_t = f_t * c_{t-1} + i_t * tanh(W_g [x_t, h_{t-1}])
    o_t = sigmoid(W_o [x_t, h_{t-1}] + p_o * c_t)
    h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_dim, units):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.gates = nn.Linear(input_dim + units, 4 * units)
        self.peephole_i = nn.Parameter(torch.empty(units).uniform_(-0.1, 0.1))
        self.peephole_f = nn.Parameter(torch.empty(units).uniform_(-0.1, 0.1))
        self.peephole_o = nn.Parameter(torch.empty(units).uniform_(-0.1, 0.1))
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for gate in self.gates.weight.chunk(4, dim=0):
                nn.init.xavier_uniform_(gate[:, :self.input_dim])
                nn.init.orthogonal_(gate[:, self.input_dim:])
            nn.init.zeros_(self.gates.bias)
            self.gates.bias[self.units:2 * self.units] = 1.0  # forget gate


    def forward(self, x_t, h_prev, c_prev):
        z = self.gates(torch.cat([x_t, h_prev], dim=-1))
        i, f, g, o_pre = z.chunk(4, dim=-1)
        i = torch.sigmoid(i + self.peephole_i * c_prev)
        f = torch.sigmoid(f + self.peephole_f * c_prev)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        o = torch.sigmoid(o_pre + self.peephole_o * c)
        h = o * torch.tanh(c)
        return h, c

    def count_parameters(self):
        """Count total learnable parameters."""
        return 4 * self.units * (self.units + self.input_dim + 1) + 3 * self.units

class BiLSTM(nn.Module):
    """Returns per-step outputs (B, T, 2 * units) and final states (2, B, units)."""

    def __init__(self, input_dim, units, cell=LSTMCell):
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
        c_fwd = torch.zeros(B, self.units, device=x.device)  # (B, H)
        fwd_outputs = []

        for t in range(T):
            valid = mask[:, t].unsqueeze(-1)  # (B, 1)
            h_new, c_new = self.fwd(x[:, t, :], h_fwd, c_fwd)
            h_fwd = torch.where(valid, h_new, h_fwd)
            c_fwd = torch.where(valid, c_new, c_fwd)
            fwd_outputs.append(h_fwd * valid)

        h_bwd = torch.zeros(B, self.units, device=x.device)
        c_bwd = torch.zeros(B, self.units, device=x.device)
        bwd_outputs = [None] * T
        for t in reversed(range(T)):
            valid = mask[:, t].unsqueeze(-1)
            h_new, c_new = self.bwd(x[:, t, :], h_bwd, c_bwd)
            h_bwd = torch.where(valid, h_new, h_bwd)
            c_bwd = torch.where(valid, c_new, c_bwd)
            bwd_outputs[t] = h_bwd * valid

        fwd_outputs = torch.stack(fwd_outputs, dim=1)  # (B, T, H)
        bwd_outputs = torch.stack(bwd_outputs, dim=1)  # (B, T, H)
        outputs = torch.cat([fwd_outputs, bwd_outputs], dim=-1)  # (B, T, 2H)
        final_states = torch.stack((h_fwd, h_bwd), dim=0)  # (2, B, H)
        return outputs, final_states


    def count_parameters(self):
        """Count total learnable parameters."""
        return self.fwd.count_parameters() + self.bwd.count_parameters()


class DeepBiLSTM(nn.Module):
    """Returns per-step outputs (B, T, 2 * units) and final states (2 * num_layers, B, units)."""

    def __init__(self, input_dim, units, num_layers, cell=LSTMCell):
        super().__init__()
        self.layers = nn.ModuleList([
            BiLSTM(input_dim if i == 0 else 2 * units, units, cell)
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


class DeepLSTM(nn.Module):
    """Returns per-step outputs (B, T, units) and final states (num_layers, B, units)."""

    def __init__(self, input_dim, units, num_layers, cell=LSTMCell):
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
            c = torch.zeros(B, self.units, device=x.device)  # (B, H)
            outputs = []

            for t in range(T):
                valid = mask[:, t].unsqueeze(-1)  # (B, 1)
                h_new, c_new = cell(x[:, t, :], h, c)
                h = torch.where(valid, h_new, h)
                c = torch.where(valid, c_new, c)
                outputs.append(h * valid)

            x = torch.stack(outputs, dim=1)  # (B, T, H)
            final_states.append(h)

        return x, torch.stack(final_states, dim=0)  # (num_layers, B, H)

    def count_parameters(self):
        """Count total learnable parameters."""
        return sum(cell.count_parameters() for cell in self.cells)
