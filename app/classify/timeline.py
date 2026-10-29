"""The classifier's KB: a staircase timeline per (subject, attribute) + page trust by EM.

Ported from X's research (eda/improve/timeline.py, eda/improve/trust.py). Each fact gets an anchor code per run:
  PIN (1)     a fact of an input row labelled true: the stairs pay ANCHOR to disagree with it
  FORBID (-1) the fact of an input row labelled false with a single fact: the stairs pay ANCHOR to hold it
  VOTE (0)    a fact of an unlabelled row: a vote weighted by its page's trust; one page votes once per value and date
  IGNORE (9)  a generated training item (a query: features, never a vote), or a labelled row we cannot pin
              (false with several facts: which one is false?; unsure)
Labels only ever come from the input's labelled rows; the caller decides which (all of them for inference,
all but one fold of pages for the out-of-fold features of the labelled rows themselves).
Changing value between consecutive dates costs SWITCH; a new stair needs a pin or >= 2 distinct voting pages.
Page trust (EM, 3 rounds): from a text-only prior (hype pages start lower),
trust = (votes that agree with the stairs + PSEUDO * prior) / (informative votes + PSEUDO).
Values compare with app.kb.normalize.key_inclusive (units converted, "up to 60" states 60), facts of one date in
row-number order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.kb import normalize as nz

SWITCH = 1.5         # cost of a value change between two consecutive mention dates (> one page's vote)
ANCHOR = 6.0         # cost to disagree with a pinned value (or to hold a forbidden one)
ANCHOR_SUPPORT = 2.0  # a pin's weight when counting support around a date
TAU = 90.0           # days: decay of a vote's weight when measuring support around a date
PSEUDO = 3.0         # pull of the prior on a page with few informative votes
ROUNDS = 3
PIN, FORBID, VOTE, IGNORE = 1, -1, 0, 9
QUERY = IGNORE
PAIR_SEP = "\x1f"    # between subject and attribute in a pair key (never in a name)
EPOCH = pd.Timestamp("2025-01-01")
FEATURES = ["tl_agree", "tl_was_held_before", "tl_held_later", "days_since_stair", "days_to_next_stair", "n_stairs",
            "support_for", "support_against", "support_margin", "n_pages_for", "n_pages_against", "pair_mentions",
            "pair_pages", "anchored_for", "forbidden"]


def day_of(date: str) -> int:
    d = str(date)[:10]
    if len(d) == 7:
        d += "-01"
    elif len(d) == 4:
        d += "-01-01"
    try:
        return int((pd.Timestamp(d) - EPOCH).days)
    except (ValueError, TypeError):
        return 0


class Timeline:
    """facts: DataFrame with row_id, subject, attribute, value (dict), date, page, conf (and any extra columns)."""

    def __init__(self, facts: pd.DataFrame):
        f = facts.reset_index(drop=True).copy()
        f["vkey"] = [nz.key_inclusive(v) for v in f["value"]]
        f["day"] = [day_of(d) for d in f["date"]]
        f["order"] = [nz.row_order(r) for r in f["row_id"]]
        f = f.sort_values(["day", "order"], kind="stable").reset_index(drop=True)
        self.f = f
        self.n = len(f)
        pair_key = f["subject"].astype(str) + PAIR_SEP + f["attribute"].astype(str)
        self.pair = pd.factorize(pair_key)[0] if self.n else np.zeros(0, int)
        self.pairs = list(dict.fromkeys(zip(f["subject"], f["attribute"])))
        codes, uniques = pd.factorize(f["vkey"]) if self.n else (np.zeros(0, int), [])
        self.val, self.val_keys = codes, list(uniques)
        self.day = f["day"].to_numpy()
        self.page = f["page"].to_numpy()
        self.conf = f["conf"].to_numpy(dtype=float)
        self.groups = [np.flatnonzero(self.pair == p) for p in range(len(self.pairs))]
        self.group_of = {pr: self.groups[p] for p, pr in enumerate(self.pairs)}
        self.value_of = dict(zip(f["vkey"], f["value"]))
        self.stairs: dict[tuple, list[tuple[int, str]]] = {}
        self.anchor = np.full(self.n, IGNORE)  # the anchors of the last run (sources, value_at)

    # ------------------------------------------------------------------ dynamic program
    def _path(self, idx, anchor, weight):
        days_u, t_of = np.unique(self.day[idx], return_inverse=True)
        vals_u, v_of = np.unique(self.val[idx], return_inverse=True)
        T, V = len(days_u), len(vals_u)
        C = np.zeros((T, V))
        seen = {}
        for k, m in enumerate(idx):
            t, j, a = t_of[k], v_of[k], anchor[m]
            if a == PIN:
                C[t] += ANCHOR
                C[t, j] -= ANCHOR
            elif a == FORBID:
                C[t, j] += ANCHOR
            elif a == VOTE:
                key = (t, j, self.page[m])
                seen[key] = max(seen.get(key, 0.0), weight[m])
        for (t, j, _), w in seen.items():
            C[t] += w
            C[t, j] -= w
        dp = C[0].copy()
        back = np.zeros((T, V), dtype=int)
        for t in range(1, T):
            best = dp.argmin()
            move = dp[best] + SWITCH
            back[t] = np.where(dp <= move, np.arange(V), best)
            dp = np.minimum(dp, move) + C[t]
        path = np.zeros(T, dtype=int)
        path[-1] = dp.argmin()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return days_u, vals_u, t_of, v_of, self._two_page_rule(idx, anchor, t_of, v_of, path)

    def _two_page_rule(self, idx, anchor, t_of, v_of, path):
        """Fold back any later stair that no pin and fewer than 2 distinct voting pages support."""
        changed = True
        while changed:
            changed = False
            starts = [0] + [t for t in range(1, len(path)) if path[t] != path[t - 1]]
            for s_i in range(1, len(starts)):
                s = starts[s_i]
                e = starts[s_i + 1] if s_i + 1 < len(starts) else len(path)
                ms = idx[(t_of >= s) & (t_of < e) & (v_of == path[s])]
                if (anchor[ms] == PIN).any() or len(set(self.page[ms[anchor[ms] == VOTE]])) >= 2:
                    continue
                path[s:e] = path[s - 1]
                changed = True
                break
        return path

    # ------------------------------------------------------------------ run + features
    def run(self, anchor, weight, scored=None):
        """Features for the facts in `scored` (all when None) and, per vote, whether it agrees with the stairs
        (page trust uses it). Stairs and anchors of this run are kept for value_at / supporters."""
        scored = np.ones(self.n, bool) if scored is None else scored
        feats = {k: np.full(self.n, np.nan) for k in FEATURES}
        vote_agree = np.full(self.n, np.nan)
        stairs = {}
        for p, idx in enumerate(self.groups):
            days_u, vals_u, t_of, v_of, path = self._path(idx, anchor, weight)
            stairs[self.pairs[p]] = [(int(days_u[t]), self.val_keys[vals_u[path[t]]])
                                     for t in range(len(path)) if t == 0 or path[t] != path[t - 1]]
            if not scored[idx].any() and not (anchor[idx] == VOTE).any():
                continue
            V = len(vals_u)
            pg, dy, an = self.page[idx], self.day[idx], anchor[idx]
            vote, pin = an == VOTE, an == PIN
            base_w = np.where(pin, ANCHOR_SUPPORT, np.where(vote, weight[idx], 0.0))
            informative = vote | pin
            n_pages_pair = len(set(pg[informative]))
            change_t = np.array([t for t in range(1, len(path)) if path[t] != path[t - 1]], dtype=int)
            pinned_val = np.zeros(V, bool)
            pinned_val[v_of[pin]] = True
            for k, m in enumerate(idx):
                t, j = t_of[k], v_of[k]
                if vote[k] and n_pages_pair >= 2:
                    vote_agree[m] = float(path[t] == j)
                if not scored[m]:
                    continue
                agree = path[t] == j
                feats["tl_agree"][m] = float(agree)
                feats["tl_was_held_before"][m] = float((not agree) and (j in path[:t]))
                feats["tl_held_later"][m] = float((not agree) and (j in path[t + 1:]))
                prev = change_t[change_t <= t]
                nxt = change_t[change_t > t]
                feats["days_since_stair"][m] = days_u[t] - days_u[prev[-1] if len(prev) else 0]
                feats["days_to_next_stair"][m] = (days_u[nxt[0]] - days_u[t]) if len(nxt) else np.nan
                feats["n_stairs"][m] = len(change_t) + 1
                feats["pair_mentions"][m] = int(informative.sum())
                feats["pair_pages"][m] = n_pages_pair
                o = informative & (pg != pg[k])
                w = base_w[o] * np.exp(-np.abs(dy[o] - dy[k]) / TAU)
                by_val = np.bincount(v_of[o], weights=w, minlength=V)
                sup_for = by_val[j]
                by_val[j] = 0.0
                forbid = bool(((an == FORBID) & (pg != pg[k]) & (v_of == j)).any())
                feats["support_for"][m] = sup_for
                feats["support_against"][m] = by_val.sum() + (ANCHOR_SUPPORT if forbid else 0.0)
                feats["support_margin"][m] = sup_for - by_val.max()
                feats["n_pages_for"][m] = len(set(pg[o & (v_of == j)]))
                feats["n_pages_against"][m] = len(set(pg[o & (v_of != j)]))
                feats["anchored_for"][m] = float(pinned_val[j] and not (pin[k] and (pin & (v_of == j)).sum() == 1))
                feats["forbidden"][m] = float(forbid)
        self.stairs, self.anchor = stairs, np.asarray(anchor)
        return pd.DataFrame(feats), vote_agree

    # ------------------------------------------------------------------ lookups on the last run's stairs
    def value_at(self, subject: str, attribute: str, date: str):
        """(vkey, value dict) of the stair in force on `date`, or (None, None)."""
        st = self.stairs.get((subject, attribute))
        if not st:
            return None, None
        d = day_of(date)
        cur = [k for day, k in st if day <= d]
        k = cur[-1] if cur else None
        return k, (self.value_of.get(k) if k else None)

    def sequence(self, subject: str, attribute: str) -> list[str]:
        return [k for _, k in self.stairs.get((subject, attribute), [])]


def page_trust(tl: Timeline, anchor, prior_by_page: dict, rounds: int = ROUNDS):
    """EM: page reliability <-> stairs. Returns (trust per page, vote weight per fact)."""
    pages = tl.page
    trust = {p: prior_by_page.get(p, 0.7) for p in set(pages)}
    weight = np.array([trust[p] for p in pages]) * tl.conf
    for _ in range(rounds):
        _, agree = tl.run(anchor, weight, scored=np.zeros(tl.n, bool))
        df = pd.DataFrame({"page": pages, "agree": agree}).dropna()
        stats = df.groupby("page")["agree"].agg(["sum", "count"])
        for p in trust:
            s, c = (stats.loc[p, "sum"], stats.loc[p, "count"]) if p in stats.index else (0.0, 0.0)
            trust[p] = float((s + PSEUDO * prior_by_page.get(p, 0.7)) / (c + PSEUDO))
        weight = np.array([trust[p] for p in pages]) * tl.conf
    return trust, weight


def text_prior(hype_share_by_page: dict) -> dict:
    """Label-free starting trust: release-note-like pages 0.8, first-person hype pages down to 0.4."""
    return {p: float(np.clip(0.8 - 0.4 * h, 0.4, 0.8)) for p, h in hype_share_by_page.items()}


def supporters(tl: Timeline, subject: str, attribute: str, value: dict, date: str, limit: int = 3):
    """Rows whose facts state the same value (support) or another value (against), labelled rows first
    (a pinned true row is the strongest source), then nearest in time."""
    idx = tl.group_of.get((subject, attribute))
    if idx is None:
        return [], []
    vk = nz.key_inclusive(value)
    d = day_of(date)
    sup, aga = [], []
    real = [m for m in idx if tl.f["kind"].iat[m] == "sentence"]
    for m in sorted(real, key=lambda m: (tl.anchor[m] != PIN, abs(tl.day[m] - d), tl.f["order"].iat[m])):
        rid = tl.f["row_id"].iat[m]
        bucket = sup if tl.f["vkey"].iat[m] == vk else aga
        if rid not in sup and rid not in aga and len(bucket) < limit:
            bucket.append(rid)
    return sup, aga
