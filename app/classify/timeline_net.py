"""Option D: a liquid time-constant network (CfC) reads each fact's timeline.

For every extracted fact, the sequence is the other pages' mentions of the
same (subject, attribute), nearest in time first, then put in date order. Each
step says how far away in time the mention is, whether it states the same
value, whether its row is a KNOWN true / false row, and how its number
compares. Times are irregular, which is what a CfC (closed-form continuous-
time RNN) is built for. A GRU runs as the control: does "liquid" beat a plain
RNN?

The net predicts P(row is false) per fact; a row takes the max over its facts.
In `TruthStack` it is a third base model next to LightGBM and LogReg, trained
inside the same out-of-fold loop, so the meta-learner sees it like a feature.

Known labels enter the sequences exactly like the KB features: training rows
only, never a row's own page.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_LEN = 24
N_STEP_FEATURES = 8


def build_sequences(F: pd.DataFrame, known_labels: dict[int, str],
                    target_rows) -> tuple[np.ndarray, np.ndarray]:
    """(sequences [n_facts, MAX_LEN, N_STEP_FEATURES], row index per fact)."""
    target = set(int(r) for r in target_rows)
    true_rows = {i for i, lab in known_labels.items() if lab == "true"}
    false_rows = {i for i, lab in known_labels.items() if lab == "false"}
    seqs, fact_rows = [], []
    for _, g in F.groupby("pair", sort=False):
        days = g["date"].to_numpy().astype("datetime64[D]").astype(float)
        pages = g["page"].to_numpy()
        vkeys = g["vkey"].to_numpy()
        vnums = g["vnum"].to_numpy(dtype=float)
        rows = g["row"].to_numpy()
        known_t = np.isin(rows, list(true_rows))
        known_f = np.isin(rows, list(false_rows))
        for j in np.flatnonzero(np.isin(rows, list(target))):
            seq = np.zeros((MAX_LEN, N_STEP_FEATURES), dtype=np.float32)
            other = np.flatnonzero(pages != pages[j])
            if len(other):
                dt = days[other] - days[j]
                pick = other[np.argsort(np.abs(dt), kind="stable")[:MAX_LEN]]
                pick = pick[np.argsort(days[pick], kind="stable")]
                dtp = (days[pick] - days[j]) / 365.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.log10(vnums[j] / vnums[pick])
                ratio = np.where(np.isfinite(ratio), np.clip(ratio, -2, 2), 0.0)
                steps = np.column_stack([
                    dtp,
                    (vkeys[pick] == vkeys[j]).astype(float),
                    known_t[pick].astype(float),
                    known_f[pick].astype(float),
                    (dtp > 0).astype(float),
                    ratio,
                    np.abs(dtp) < 30 / 365.0,
                    np.ones(len(pick)),
                ])
                seq[MAX_LEN - len(pick):] = steps  # left-pad: last step = latest mention
            seqs.append(seq)
            fact_rows.append(rows[j])
    if not seqs:
        return np.zeros((0, MAX_LEN, N_STEP_FEATURES), np.float32), np.zeros(0, int)
    return np.stack(seqs), np.asarray(fact_rows)


def _net(kind: str, units: int):
    import torch

    class TimelineNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.units = units
            if kind == "cfc":
                from ncps.torch import CfCCell
                self.cell = CfCCell(N_STEP_FEATURES, units)
            else:
                self.rnn = torch.nn.GRU(N_STEP_FEATURES, units, batch_first=True)
            self.head = torch.nn.Linear(units, 1)

        def forward(self, x):
            if kind == "cfc":
                # The CfC state evolves over the real time between mentions (in
                # months; padding steps get one unit). Stepped by hand because
                # ncps.CfC squeezes per-sample timespans to the wrong shape.
                gap = torch.diff(x[:, :, 0], dim=1, prepend=x[:, :1, 0]).abs() * 12.0
                ts = torch.where(x[:, :, 7] > 0, gap.clamp(min=0.05), torch.ones_like(gap))
                h = torch.zeros(x.shape[0], self.units)
                for t in range(x.shape[1]):
                    _, h = self.cell(x[:, t], h, ts[:, t:t + 1])
                return self.head(h).squeeze(-1)
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return TimelineNet()


@dataclass
class SequenceModel:
    """CfC or GRU over fact timelines -> P(row is false) per fact."""

    kind: str = "cfc"
    units: int = 16
    epochs: int = 60
    lr: float = 3e-3
    seed: int = 0
    budget_s: float = 90.0
    state: dict = field(default_factory=dict)

    def fit(self, seqs: np.ndarray, y: np.ndarray) -> "SequenceModel":
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        net = _net(self.kind, self.units)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        x = torch.tensor(seqs)
        t = torch.tensor(y, dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.seed)
        start = time.time()
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(t), generator=gen)
            for b in range(0, len(t), 128):
                idx = perm[b:b + 128]
                opt.zero_grad()
                loss_fn(net(x[idx]), t[idx]).backward()
                opt.step()
            if time.time() - start > self.budget_s:
                break
        self.state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        return self

    def predict(self, seqs: np.ndarray) -> np.ndarray:
        import torch
        if len(seqs) == 0:
            return np.zeros(0)
        torch.set_num_threads(2)
        net = _net(self.kind, self.units)
        net.load_state_dict(self.state)
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(seqs))).numpy()


def row_scores(p_fact: np.ndarray, fact_rows: np.ndarray, rows) -> np.ndarray:
    """Max P(false) over each row's facts; NaN for rows without facts."""
    s = pd.Series(p_fact).groupby(np.asarray(fact_rows)).max()
    return pd.Series(np.asarray(rows)).map(s).to_numpy(dtype=float)


def fact_labels(fact_rows: np.ndarray, labels_by_row: dict[int, str]) -> np.ndarray:
    return np.asarray([float(labels_by_row[int(r)] == "false") for r in fact_rows])


def safe_logit(p: np.ndarray, fill: float) -> np.ndarray:
    p = np.where(np.isnan(p), fill, p)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


