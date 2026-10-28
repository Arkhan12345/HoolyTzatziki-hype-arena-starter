"""Evidence-sum judge: each piece of evidence moves the log-odds of 'false' by a Beta-smoothed likelihood ratio.

Evidence items (categorical, label-free per row): the row's archetype (eda/03_recommend.py rules evaluated on
the fold's features), its worst fact status, its page-trust band, its three-valued state and its joke tail.
log O(false) = log prior odds + sum_i log [ P(item_i | false) / P(item_i | not false) ]
Temperature scaling then fits one T on out-of-fold probabilities (minimum log-loss).
"""
import numpy as np
import pandas as pd

ALPHA = 1.0


class EvidenceJudge:
    def fit(self, items: pd.DataFrame, is_false: np.ndarray):
        y = np.asarray(is_false).astype(bool)
        self.prior = np.log((y.sum() + ALPHA) / ((~y).sum() + ALPHA))
        self.tables = {}
        for c in items.columns:
            v = items[c].astype(str)
            cats = sorted(v.unique())
            k = len(cats)
            pf = (v[y].value_counts().reindex(cats, fill_value=0) + ALPHA) / (y.sum() + ALPHA * k)
            pn = (v[~y].value_counts().reindex(cats, fill_value=0) + ALPHA) / ((~y).sum() + ALPHA * k)
            self.tables[c] = np.log(pf / pn).to_dict()
        return self

    def log_odds(self, items: pd.DataFrame):
        lo = np.full(len(items), self.prior)
        for c, tab in self.tables.items():
            lo += items[c].astype(str).map(tab).fillna(0.0).to_numpy()
        return lo


def fit_temperature(proba, y, grid=np.linspace(0.5, 3.0, 26)):
    logp = np.log(np.clip(proba, 1e-9, 1))
    best, best_t = np.inf, 1.0
    for t in grid:
        q = np.exp(logp / t)
        q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-9, 1)).mean()
        if nll < best:
            best, best_t = nll, t
    return best_t


def apply_temperature(proba, t):
    q = np.exp(np.log(np.clip(proba, 1e-9, 1)) / t)
    return q / q.sum(1, keepdims=True)


def with_false_prob(proba, p_false):
    """Replace P(false) and rescale true/unsure to keep their ratio."""
    rest = proba[:, 1:].sum(1, keepdims=True)
    share = np.where(rest > 0, proba[:, 1:] / np.maximum(rest, 1e-12), [[1.0, 0.0]])
    return np.column_stack([p_false, (1 - p_false)[:, None] * share])
