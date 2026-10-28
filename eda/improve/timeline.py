"""Staircase timeline: the value each (subject, attribute) holds on each date, from every mention.

Per pair, a sticky dynamic program picks one value per mention date:
  - a labelled TRUE row pins its value (ANCHOR cost to disagree);
  - a labelled FALSE single-fact row forbids its value (ANCHOR cost to hold it);
  - every unlabelled mention is a vote, weighted by its page's trust; one page votes once per value and date;
  - changing value between consecutive dates costs SWITCH.
Our rule on top: a new stair must be backed by >= 2 distinct pages (or a labelled anchor); a stair one page
props up alone is folded back into the stair before it.

Features for a scored fact (votes are weighted by exp(-|days| / TAU) around its date, own page excluded):
  tl_agree, tl_was_held_before (outdated), tl_held_later (ahead of time), days_since_stair, days_to_next_stair,
  n_stairs, support_for, support_against, support_margin, n_pages_for, n_pages_against, anchored_for, forbidden.
"""
import numpy as np
import pandas as pd

ANCHOR = 6.0     # cost to disagree with a labelled true row (or to hold a labelled-false value)
SWITCH = 1.5     # cost of a value change between two consecutive mention dates (> one page's vote)
TAU = 90.0       # days: decay of a vote's weight when measuring support around a date
ANCHOR_SUPPORT = 2.0


class Timeline:
    """Build once per fact subset; run() many times with different anchors / weights."""

    def __init__(self, facts):
        f = facts.reset_index(drop=True)
        self.f = f
        self.n = len(f)
        self.pair = pd.factorize(list(zip(f["subject"], f["attribute"])))[0]
        self.val = pd.factorize(f["vkey"])[0]
        self.day = ((f["date"] - pd.Timestamp("2025-01-01")).dt.days).to_numpy()
        self.page = f["page"].to_numpy()
        self.row = f["id"].to_numpy()
        self.groups = [np.flatnonzero(self.pair == p) for p in range(self.pair.max() + 1)]

    # ------------------------------------------------------------------ dynamic program
    def _path(self, idx, anchor, weight):
        days_u, t_of = np.unique(self.day[idx], return_inverse=True)
        vals_u, v_of = np.unique(self.val[idx], return_inverse=True)
        T, V = len(days_u), len(vals_u)
        C = np.zeros((T, V))
        seen = {}
        for k, m in enumerate(idx):
            t, j, a = t_of[k], v_of[k], anchor[m]
            if a == 1:
                C[t] += ANCHOR
                C[t, j] -= ANCHOR
            elif a == -1:
                C[t, j] += ANCHOR
            elif a == 0:
                key = (t, j, self.page[m])
                seen[key] = max(seen.get(key, 0.0), weight[m])
        for (t, j, _), w in seen.items():
            C[t] += w
            C[t, j] -= w
        dp = C[0].copy()
        back = np.zeros((T, V), dtype=int)
        for t in range(1, T):
            best = dp.argmin()
            stay = dp
            move = dp[best] + SWITCH
            back[t] = np.where(stay <= move, np.arange(V), best)
            dp = np.minimum(stay, move) + C[t]
        path = np.zeros(T, dtype=int)
        path[-1] = dp.argmin()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        path = self._two_page_rule(idx, anchor, t_of, v_of, path)
        return days_u, vals_u, t_of, v_of, path

    def _two_page_rule(self, idx, anchor, t_of, v_of, path):
        """Fold back any later stair that fewer than 2 distinct pages (and no anchor) support."""
        changed = True
        while changed:
            changed = False
            starts = [0] + [t for t in range(1, len(path)) if path[t] != path[t - 1]]
            for s_i in range(1, len(starts)):
                s = starts[s_i]
                e = starts[s_i + 1] if s_i + 1 < len(starts) else len(path)
                v = path[s]
                inside = (t_of >= s) & (t_of < e) & (v_of == v)
                ms = idx[inside]
                if (anchor[ms] == 1).any() or len(set(self.page[ms[anchor[ms] == 0]])) >= 2:
                    continue
                path[s:e] = path[s - 1]
                changed = True
                break
        return path

    # ------------------------------------------------------------------ run + features
    def run(self, anchor, weight, scored):
        """anchor: int per fact (1 true, -1 false single-fact, 0 vote, 9 ignore); weight: vote weight per fact;
        scored: bool mask of facts to return features for. Returns (features DataFrame indexed like facts,
        agree array for every vote, used by page trust)."""
        feats = {k: np.full(self.n, np.nan) for k in
                 ["tl_agree", "tl_was_held_before", "tl_held_later", "days_since_stair", "days_to_next_stair",
                  "n_stairs", "support_for", "support_against", "support_margin", "n_pages_for",
                  "n_pages_against", "anchored_for", "forbidden", "pair_mentions"]}
        vote_agree = np.full(self.n, np.nan)
        for idx in self.groups:
            days_u, vals_u, t_of, v_of, path = self._path(idx, anchor, weight)
            V = len(vals_u)
            pg, dy, an = self.page[idx], self.day[idx], anchor[idx]
            base_w = np.where(an == 1, ANCHOR_SUPPORT, np.where(an == 0, weight[idx], 0.0))
            n_pages_pair = len(set(pg))
            change_t = np.array([t for t in range(1, len(path)) if path[t] != path[t - 1]], dtype=int)
            anchored_val = np.zeros(V, bool)
            anchored_val[v_of[an == 1]] = True
            for k, m in enumerate(idx):
                t, j = t_of[k], v_of[k]
                if an[k] == 0 and n_pages_pair >= 2:
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
                feats["pair_mentions"][m] = len(idx)
                o = pg != pg[k]
                w = base_w[o] * np.exp(-np.abs(dy[o] - dy[k]) / TAU)
                by_val = np.bincount(v_of[o], weights=w, minlength=V)
                sup_for = by_val[j]
                by_val[j] = 0.0
                forbid = bool(((an[o] == -1) & (v_of[o] == j)).any())
                feats["support_for"][m] = sup_for
                feats["support_against"][m] = by_val.sum() + (ANCHOR_SUPPORT if forbid else 0.0)
                feats["support_margin"][m] = sup_for - by_val.max()
                voting = o & (an != 9)
                feats["n_pages_for"][m] = len(set(pg[voting & (v_of == j)]))
                feats["n_pages_against"][m] = len(set(pg[voting & (v_of != j)]))
                feats["anchored_for"][m] = float(anchored_val[j])
                feats["forbidden"][m] = float(forbid)
        return pd.DataFrame(feats), vote_agree


def anchors_for(timeline, labelled_ids, labels_by_id, nfacts_by_id):
    """Anchor code per fact: labelled true -> 1, labelled false single-fact -> -1, labelled false multi-fact
    -> 9 (ignored: we don't know which fact is the false one), labelled unsure -> 9, unlabelled -> 0 (vote)."""
    a = np.zeros(timeline.n, dtype=int)
    for m, rid in enumerate(timeline.row):
        if rid in labelled_ids:
            lab = labels_by_id[rid]
            if lab == "true":
                a[m] = 1
            elif lab == "false" and nfacts_by_id[rid] == 1:
                a[m] = -1
            else:
                a[m] = 9
    return a
