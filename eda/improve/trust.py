"""Page trust by EM: alternate page reliability <-> staircase timeline (one-coin Dawid-Skene style, numpy).

Start: every page gets the prior of its genre (the true-rate of labelled rows of that genre, Beta-smoothed
toward the overall rate), so a first-person hype page starts less trusted than release notes.
Round: run the timeline with votes weighted by trust; a page's trust = (agreeing votes + a * prior) /
(informative votes + a), where a vote is informative when its pair is mentioned by >= 2 pages.
"""
import numpy as np
import pandas as pd

PSEUDO = 3.0          # pull of the genre prior on a page with few informative votes
GENRE_PSEUDO = 10.0   # pull of the overall rate on a genre with few labelled rows
ROUNDS = 3


def genre_priors(rows, labelled_ids, genre_by_id):
    lab = rows[rows["id"].isin(labelled_ids)]
    is_true = (lab["label"] == "true").astype(float)
    overall = is_true.mean()
    g = lab["id"].map(genre_by_id)
    stats = is_true.groupby(g).agg(["sum", "count"])
    prior = (stats["sum"] + GENRE_PSEUDO * overall) / (stats["count"] + GENRE_PSEUDO * 1.0)
    return prior.to_dict(), overall


def page_trust(timeline, anchor, prior_by_page, rounds=ROUNDS):
    """Returns (trust per page dict, vote weight per fact)."""
    pages = timeline.page
    trust = {p: prior_by_page[p] for p in set(pages)}
    weight = np.array([trust[p] for p in pages])
    none_scored = np.zeros(timeline.n, dtype=bool)
    for _ in range(rounds):
        _, agree = timeline.run(anchor, weight, none_scored)
        df = pd.DataFrame({"page": pages, "agree": agree}).dropna()
        stats = df.groupby("page")["agree"].agg(["sum", "count"])
        for p in trust:
            s, c = (stats.loc[p, "sum"], stats.loc[p, "count"]) if p in stats.index else (0.0, 0.0)
            trust[p] = (s + PSEUDO * prior_by_page[p]) / (c + PSEUDO)
        weight = np.array([trust[p] for p in pages])
    return trust, weight
