"""Row-level networks for the neural track: a recurrent cell over the row's corpus timeline + tabular features."""
import numpy as np
import torch


def encode_dt(seq, dt):
    """seq[..., 0] = day gap to the row (years), seq[..., 3] = gap between mentions (years). Returns seq, timespans."""
    s = seq.copy()
    if dt == "log":
        for c in (0, 3):
            s[..., c] = np.sign(s[..., c]) * np.log1p(np.abs(s[..., c]) * 365.0) / np.log1p(365.0)
    elif dt == "none":
        s[..., 0] = 0.0
        s[..., 3] = 0.0
    ts = (np.abs(seq[..., 3]) * 365.0 / 30.0 + 1.0).astype(np.float32) if dt != "none" else \
        np.ones(seq.shape[:2], np.float32)
    return s.astype(np.float32), ts


def liquid_last(rnn, s, ts):
    """Run an ncps CfC/LTC over s [batch, T, f] with per-step time gaps ts [batch, T]; return the last state.
    Steps the cell directly: ncps 1.0.1 squeezes the gaps to [batch], which cannot broadcast against the state."""
    cell = rnn.rnn_cell
    h = torch.zeros(s.shape[0], rnn.state_size, device=s.device)
    out = None
    for t in range(s.shape[1]):
        out, h = cell(s[:, t], h, ts[:, t:t + 1])
    return out


class RowNet(torch.nn.Module):
    """cell in {cfc, ltc, gru}; liquid cells get the real time gaps as timespans."""

    def __init__(self, cell, n_seq, n_tab, hidden=32, n_out=3):
        super().__init__()
        self.cell = cell
        if cell == "cfc":
            from ncps.torch import CfC
            self.rnn = CfC(n_seq, hidden, batch_first=True)
        elif cell == "ltc":
            from ncps.torch import LTC
            self.rnn = LTC(n_seq, hidden, batch_first=True)
        else:
            self.rnn = torch.nn.GRU(n_seq, hidden, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden + n_tab, 48), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(48, n_out))

    def forward(self, s, ts, x):
        if self.cell == "gru":
            _, h = self.rnn(s)
            h = h[-1]
        else:
            h = liquid_last(self.rnn, s, ts)
        return self.head(torch.cat([h, x], 1))


class SpikingHead(torch.nn.Module):
    """Leaky integrate-and-fire MLP with rate coding (inputs in [0, 1]), any number of outputs."""

    def __init__(self, n_in, hidden=64, steps=25, n_out=2):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        sg = surrogate.fast_sigmoid(slope=25)
        self.steps = steps
        self.fc1, self.lif1 = torch.nn.Linear(n_in, hidden), snn.Leaky(beta=0.9, spike_grad=sg, init_hidden=False)
        self.fc2, self.lif2 = torch.nn.Linear(hidden, hidden), snn.Leaky(beta=0.9, spike_grad=sg, init_hidden=False)
        self.fc3 = torch.nn.Linear(hidden, n_out)

    def forward(self, x):
        m1 = torch.zeros(x.shape[0], self.fc1.out_features, device=x.device)
        m2 = torch.zeros(x.shape[0], self.fc2.out_features, device=x.device)
        acc = 0
        for _ in range(self.steps):
            spk_in = (torch.rand_like(x) < x).float()  # rate coding
            s1, m1 = self.lif1(self.fc1(spk_in), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            acc = acc + self.fc3(s2)  # readout = summed membrane drive of spikes
        return acc / self.steps


class CharBiGRU(torch.nn.Module):
    def __init__(self, vocab, emb=32, hidden=64, n_out=3):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = torch.nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(4 * hidden, n_out))

    def forward(self, t):
        h, _ = self.rnn(self.emb(t))
        mask = (t > 0).unsqueeze(-1).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = (h - 1e4 * (1 - mask)).max(1).values
        return self.head(torch.cat([mean, mx], 1))


class CharCNN(torch.nn.Module):
    def __init__(self, vocab, emb=32, filters=64, n_out=3):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(emb, filters, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(3 * filters, n_out))

    def forward(self, t):
        e = self.emb(t).transpose(1, 2)
        return self.head(torch.cat([torch.relu(c(e)).max(2).values for c in self.convs], 1))


def char_encode(texts, max_len=200):
    chars = sorted({c for s in texts for c in s})
    vocab = {c: i + 2 for i, c in enumerate(chars)}  # 0 pad, 1 unknown
    arr = np.zeros((len(texts), max_len), np.int64)
    for n, s in enumerate(texts):
        ids = [vocab.get(c, 1) for c in s[:max_len]]
        arr[n, :len(ids)] = ids
    return arr, len(vocab) + 2
