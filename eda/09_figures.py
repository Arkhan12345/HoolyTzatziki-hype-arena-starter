"""Step 9: figure library for the pitch -> pitch/screenshots/<category>/ (+ INDEX.md with insight scores).

Every figure is written four times: <name>.png / <name>.svg (light) and <name>-dark.png / <name>-dark.svg.
The title states the takeaway; a small caption names the data source. Colours follow one fixed palette:
labels are always true = blue, false = orange, unsure = aqua; "before" is grey and "after" is blue.

Nothing here is wired into pitch/PITCH.md: INDEX.md is a menu to choose from.

Run:  .venv/Scripts/python eda/09_figures.py            (about 2 minutes; caches the OOF feature table)
      .venv/Scripts/python eda/09_figures.py --fresh    (rebuild the cache)
"""
import importlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
SHOTS = ROOT / "pitch" / "screenshots"
sys.path.insert(0, str(Path(__file__).parent))
CACHE = OUT / "fig_cache.pkl"

# ------------------------------------------------------------------ theme (validated default palette)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                     "#6250d6", "#e34948"], neutral="#f0efec", red="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                                    "#9085e9", "#e66767"], neutral="#383835", red="#e66767"),
}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABEL_SLOT = {"true": 0, "false": 1, "unsure": 2}
INDEX = []


def base_rc(t):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 14,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.facecolor": t["surface"],
        "figure.facecolor": t["surface"], "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "legend.frameon": False, "legend.labelcolor": t["ink2"]})


def caption(fig, t, text):
    fig.text(0.01, 0.01, f"Source: {text}", fontsize=8, color=t["muted"], ha="left", va="bottom")


def figure(cat, name, takeaway, source, scores, size=(10, 5.6)):
    """Decorator: draw(fig, t) is called once per theme; files are saved; the figure is registered."""
    def wrap(draw):
        for mode in ("light", "dark"):
            t = THEMES[mode]
            base_rc(t)
            fig = plt.figure(figsize=size)
            draw(fig, t)
            if fig.axes and fig.axes[0].get_position().width < 0.99:  # not the full-bleed card
                try:
                    fig.tight_layout(rect=(0, 0.05, 1, 1))
                except Exception:
                    pass
            caption(fig, t, source)
            d = SHOTS / cat
            d.mkdir(parents=True, exist_ok=True)
            suffix = "" if mode == "light" else "-dark"
            for ext in ("png", "svg"):
                fig.savefig(d / f"{name}{suffix}.{ext}", dpi=150, bbox_inches="tight")
            plt.close(fig)
        s, c, r = scores
        INDEX.append(dict(path=f"{cat}/{name}.png", cat=cat, takeaway=takeaway, surprise=s, clarity=c,
                          relevance=r, score=round((s * c * r) ** (1 / 3), 1), source=source))
        print(f"  {cat}/{name}")
        return draw
    return wrap


def hbar_labels(ax, bars, fmt, t, inside=False):
    for b in bars:
        w = b.get_width()
        ax.text(w + (0 if inside else ax.get_xlim()[1] * 0.01), b.get_y() + b.get_height() / 2, fmt(w),
                va="center", ha="left", fontsize=10, color=t["ink2"])


# ------------------------------------------------------------------ data
def load():
    D = {}
    D["R"] = pd.read_pickle(OUT / "features.pkl")
    D["F"] = pd.read_pickle(OUT / "facts.pkl")
    D["flags"] = pd.read_pickle(OUT / "flags.pkl")
    D["rules"] = pd.read_csv(OUT / "rules.csv")
    D["ladder"] = pd.read_csv(OUT / "improve_labels.csv")
    D["oof"] = pickle.load(open(OUT / "improve_oof.pkl", "rb"))
    D["neuro"] = pd.read_csv(OUT / "improve_neuro.csv")
    D["sweep"] = pd.read_csv(OUT / "neuro_sweep_seeds.csv")
    D["extr"] = pd.read_csv(OUT / "extraction.csv")
    D["conf"] = pd.read_csv(OUT / "confusable_attributes.csv")
    D["ann"] = pd.read_csv(OUT / "practice_annotated.csv").set_index("id")
    D["splits"] = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    D["tlx"] = json.loads((OUT / "handoff" / "timeline_examples.json").read_text(encoding="utf-8"))
    return D


def oof_feature_table(fresh=False):
    """Out-of-fold feature table (fold_group test rows) from the ladder code; cached."""
    if CACHE.exists() and not fresh:
        return pickle.load(open(CACHE, "rb"))
    L = importlib.import_module("07_improve")
    cv_ids = list(L.SPLITS.index[L.SPLITS.holdout == 0])
    parts = []
    for fi, (train, test, inner) in enumerate(L.scheme_folds("fold_group", cv_ids)):
        X, _ = L.fold_table(L.F_ALL, train, test, inner)
        parts.append(X.loc[sorted(test)])
        print(f"    oof features fold {fi}", flush=True)
    X = pd.concat(parts)
    items = L.evidence_items(X)
    pickle.dump((X, items), open(CACHE, "wb"))
    return X, items


def ladder_module():
    return importlib.import_module("07_improve")


def mf1(y, p):
    return f1_score(y, p, labels=[1, 0], average="macro", zero_division=0)


def page_boot(D, scheme, variant, base="L0_base", B=1000, seed=0):
    """Absolute macro-F1 CI and paired gain CI vs base, resampling pages."""
    o = D["oof"][scheme]
    A, V = o[base], o[variant]
    d = pd.DataFrame({"y": A["y"], "pa": A["pred"]}, index=A["ids"]).join(
        pd.DataFrame({"pb": V["pred"]}, index=V["ids"]), how="inner")
    d["page"] = D["splits"].loc[d.index, "page"].values
    pages = d.page.unique()
    idx = {p: np.where(d.page.values == p)[0] for p in pages}
    y, pa, pb = d.y.values, d.pa.values, d.pb.values
    rng = np.random.default_rng(seed)
    absb, gain = [], []
    for _ in range(B):
        ii = np.concatenate([idx[p] for p in rng.choice(pages, len(pages))])
        fb = mf1(y[ii], pb[ii])
        absb.append(fb)
        gain.append(fb - mf1(y[ii], pa[ii]))
    return (mf1(y, pb), *np.percentile(absb, [2.5, 97.5]), mf1(y, pb) - mf1(y, pa), *np.percentile(gain, [2.5, 97.5]))


# ------------------------------------------------------------------ 01 data
def figs_data(D):
    R = D["R"]
    n = R.label.value_counts()

    @figure("01_data", "label_mix", "83% of practice rows are true: a model must beat 'all true' to matter",
            "data/practice.csv (1,897 rows)", (2, 5, 5), size=(9, 3.6))
    def _(fig, t):
        ax = fig.add_subplot()
        order = ["true", "false", "unsure"]
        vals = [n[k] for k in order]
        bars = ax.barh(order[::-1], vals[::-1], color=[t["s"][LABEL_SLOT[k]] for k in order[::-1]],
                       height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.set_xlim(0, max(vals) * 1.18)
        for b, k in zip(bars, order[::-1]):
            ax.text(b.get_width() + 15, b.get_y() + b.get_height() / 2,
                    f"{n[k]:,}  ({n[k] / len(R):.1%})", va="center", color=t["ink2"])
        ax.set_title("83% true, 15% false, 2% unsure")
        ax.set_xlabel("rows")
        ax.grid(axis="y", visible=False)

    sizes = R.groupby("page").size()

    @figure("01_data", "rows_per_page", "The data is 291 pages: sentences of one page share a source and a date",
            "eda/out/rows.pkl (page = contiguous ids with one date)", (3, 4, 3))
    def _(fig, t):
        ax = fig.add_subplot()
        c = sizes.clip(upper=20).value_counts().sort_index()
        ax.bar(c.index, c.values, color=t["s"][0], width=0.8, edgecolor=t["surface"], linewidth=2)
        ax.set_title(f"291 pages over 159 dates; a page holds {sizes.median():.0f} rows (median)")
        ax.set_xlabel("rows per page (20 = 20 or more)")
        ax.set_ylabel("pages")
        ax.grid(axis="x", visible=False)

    F = D["F"]
    kr = F.groupby("kind").label.agg(n="size", false=lambda s: (s == "false").mean()).sort_values("false")
    base = (R.label == "false").mean()

    @figure("01_data", "false_rate_by_kind", "Numbers lie most: quantities are false 17% of the time, versions 7%",
            "eda/out/facts.pkl (2,402 facts)", (3, 4, 3))
    def _(fig, t):
        ax = fig.add_subplot()
        bars = ax.barh(kr.index, kr["false"], color=t["s"][1], height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.axvline(base, color=t["ink2"], lw=1.5, ls="--")
        ax.text(base, len(kr) - 0.3, f" all rows {base:.1%}", color=t["ink2"], fontsize=9, va="bottom")
        ax.set_xlim(0, kr["false"].max() * 1.3)
        for b, (k, r) in zip(bars, kr.iterrows()):
            ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2, f"{r['false']:.0%}  (n={r['n']})",
                    va="center", fontsize=9, color=t["ink2"])
        ax.set_title("False rate by value kind")
        ax.set_xlabel("share of facts labelled false")
        ax.grid(axis="y", visible=False)


# ------------------------------------------------------------------ 02 rules
def figs_rules(D):
    ex = {e["row_id"]: e for e in D["tlx"]}
    e_t, e_f = ex["p-1459"], ex["p-1474"]

    @figure("02_rules", "fake_strike", "Same sentence, two dates: true while the value holds, false once it moved",
            "eda/out/handoff/timeline_examples.json (gold timeline, Hooly Gemini median response time)",
            (5, 5, 5))
    def _(fig, t):
        ax = fig.add_subplot()
        steps = e_f["held_values"]
        xs = [pd.Timestamp(s["date"]) for s in steps] + [pd.Timestamp("2026-10-28")]
        ys = [float(s["value"]) for s in steps]
        for i in range(len(ys)):
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i]], color=t["muted"], lw=2)
            if i + 1 < len(ys):
                ax.plot([xs[i + 1], xs[i + 1]], [ys[i], ys[i + 1]], color=t["muted"], lw=1, ls=":")
            ax.text(xs[i], ys[i] + 0.05, f"held: {ys[i]:g} s ({steps[i]['pages']} pages)", color=t["ink2"],
                    fontsize=9, va="bottom")
        for e, lab in [(e_t, "true"), (e_f, "false")]:
            ax.plot(pd.Timestamp(e.get("date", None) or ("2025-11-14" if lab == "true" else "2026-08-20")),
                    float(e["sentence_value"]), "o", ms=11, color=t["s"][LABEL_SLOT[lab]],
                    markeredgecolor=t["surface"], markeredgewidth=2, zorder=5)
        ax.annotate('"replies in 2.1 s"  →  TRUE', (pd.Timestamp("2025-11-14"), 2.1), xytext=(10, -28),
                    textcoords="offset points", color=t["s"][0], fontweight="bold")
        ax.annotate('same sentence  →  FALSE', (pd.Timestamp("2026-08-20"), 2.1), xytext=(-60, 14),
                    textcoords="offset points", color=t["s"][1], fontweight="bold")
        ax.set_ylim(min(ys) - 0.5, max(ys) + 0.6)
        ax.set_title("The date decides truth: the 'fake strike'")
        ax.set_ylabel("median response time (s)")

    r = D["rules"]
    r = r[(r.confirmed) & (r.support >= 15) & (~r.family.isin(["base", "kind", "page"]))].copy()
    r["lo"] = r.false_ci95.str.extract(r"\[([\d.]+)").astype(float)
    r["hi"] = r.false_ci95.str.extract(r", ([\d.]+)\]").astype(float)
    top = pd.concat([r.nlargest(8, "p_false"), r.nsmallest(4, "p_false")]).drop_duplicates("rule")
    top = top.sort_values("p_false")
    base = 0.146

    @figure("02_rules", "rule_false_rates", "Timeline rules dominate: an outdated value is false 84% of the time",
            "eda/out/rules.csv (rules holding on ≥ 2 pages, ≥ 15 rows; 95% Wilson CI)", (4, 4, 5), size=(11, 6))
    def _(fig, t):
        ax = fig.add_subplot()
        y = np.arange(len(top))
        ax.barh(y, top.p_false, color=t["s"][1], height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.errorbar(top.p_false, y, xerr=[top.p_false - top.lo, top.hi - top.p_false], fmt="none",
                    ecolor=t["ink2"], elinewidth=1.2, capsize=3)
        ax.axvline(base, color=t["ink2"], lw=1.5, ls="--")
        ax.text(base, len(top) - 0.4, f" base rate {base:.1%}", color=t["ink2"], fontsize=9)
        ax.set_yticks(y, [s[:58] for s in top.rule], fontsize=9, color=t["ink2"])
        for yi, (p, hi, s) in enumerate(zip(top.p_false, top.hi, top.support)):
            ax.text(hi + 0.015, yi, f"{p:.0%}  n={s}", va="center", fontsize=9, color=t["ink2"])
        ax.set_xlim(0, 1.12)
        ax.set_title("P(false) when a rule fires")
        ax.set_xlabel("share of rows labelled false")
        ax.grid(axis="y", visible=False)

    X = D["flags"]
    combos = {
        "joke tail, one fact": (X.joke_tail == 1) & (X.multi_fact == 0),
        "pair has one value in corpus": (X.pair_single_value == 1) & (X.joke_tail == 0),
        "value seen before anything is known": (X.rel_before_first_known == 1) & (X.joke_tail == 0)
                                               & (X.pair_single_value == 0),
        "everything else": (X.joke_tail == 0) & (X.pair_single_value == 0) & (X.rel_before_first_known == 0),
    }
    rows = []
    for k, m in combos.items():
        s = X[m]
        rows.append((k, len(s), (s.label == "unsure").mean(), (s.label == "false").mean()))
    C = pd.DataFrame(rows, columns=["combo", "n", "unsure", "false"])

    @figure("02_rules", "unsure_combinations", "Unsure comes from jokes and unverifiable one-off values, not from lies",
            "eda/out/flags.pkl (decision-tree leaves, docs/RULES.md §8a)", (4, 4, 4))
    def _(fig, t):
        ax = fig.add_subplot()
        y = np.arange(len(C))
        h = 0.36
        b1 = ax.barh(y + h / 2, C.unsure, h, color=t["s"][2], label="unsure", edgecolor=t["surface"], linewidth=2)
        b2 = ax.barh(y - h / 2, C["false"], h, color=t["s"][1], label="false", edgecolor=t["surface"], linewidth=2)
        for b, v in list(zip(b1, C.unsure)) + list(zip(b2, C["false"])):
            ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=9,
                    color=t["ink2"])
        ax.set_yticks(y, [f"{c}  (n={n})" for c, n in zip(C.combo, C.n)], color=t["ink2"])
        ax.set_title("Where unsure lives")
        ax.set_xlabel("share of rows")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)


# ------------------------------------------------------------------ 03 method
def figs_method(D, X, items):
    L = ladder_module()
    from improve.timeline import ANCHOR, ANCHOR_SUPPORT, SWITCH, TAU, Timeline, anchors_for
    from improve.trust import GENRE_PSEUDO, PSEUDO, ROUNDS, genre_priors, page_trust
    F = L.F_ALL
    labelled = set(L.R["id"])
    tl = Timeline(F)
    anchor = anchors_for(tl, labelled, L.LABELS, L.NFACTS)
    # illustration only: every labelled row is an anchor (true pins, false forbids); no evaluation here
    pri, overall = genre_priors(L.R, labelled, L.GENRE)
    page_genre = {L.PAGE_OF[i]: L.GENRE[i] for i in F["id"].unique()}
    prior_by_page = {p: pri.get(g, overall) for p, g in page_genre.items()}
    trust, w = page_trust(tl, anchor, prior_by_page)
    fnum = pd.to_numeric(F["value_raw"], errors="coerce").to_numpy()
    lab = F["label"].to_numpy()

    def staircase(subject, attr, unit, name, takeaway, scores):
        idx = np.flatnonzero((F["subject"].to_numpy() == subject) & (F["attribute"].to_numpy() == attr))
        g = np.flatnonzero(tl.pair == tl.pair[idx[0]])
        days_u, vals_u, t_of, v_of, path = tl._path(g, anchor, w)
        val_num = {}
        for k, m in enumerate(g):
            val_num[v_of[k]] = fnum[m]
        dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(days_u, unit="D")

        @figure("03_method", name, takeaway,
                f"eda/out/facts.pkl + eda/improve/timeline.py (ANCHOR={ANCHOR}, SWITCH={SWITCH}, TAU={TAU} d, "
                "two-page rule; votes weighted by page trust)", scores)
        def _(fig, t):
            ax = fig.add_subplot()
            yv = [val_num[p] for p in path]
            ax.step(dates, yv, where="post", color=t["muted"], lw=2.5, label="held value (staircase)", zorder=2)
            for labk in ["true", "false", "unsure"]:
                m = [k for k, mm in enumerate(g) if lab[mm] == labk]
                if not m:
                    continue
                d = pd.Timestamp("2025-01-01") + pd.to_timedelta(tl.day[g[m]], unit="D")
                sz = np.array([w[g[k]] for k in m]) * 140 + 20
                ax.scatter(d, fnum[g[m]], s=sz, color=t["s"][LABEL_SLOT[labk]], edgecolor=t["surface"],
                           linewidth=1.5, zorder=4, label=f"mention labelled {labk}")
            n_changes = int((np.diff(path) != 0).sum())
            ax.set_title(f"{subject} · {attr.replace('_', ' ')}: {n_changes} value change(s) across {len(g)} mentions")
            ax.set_ylabel(unit)
            ax.legend(loc="best", fontsize=9, title="dot size = page trust", title_fontsize=8.5)

    staircase("Hooly Gemini", "median_response_time", "seconds", "staircase_gemini_response",
              "One lying page can't move the staircase: the value changes only when ≥ 2 pages agree", (4, 4, 5))
    staircase("Hooly AI Pro", "price", "EUR / month", "staircase_ai_pro_price",
              "Prices change over the year: each mention is judged against the price held on its date", (3, 4, 4))
    staircase("Hooly Search", "monthly_active_users", "users", "staircase_search_users",
              "Old numbers resurface on later pages and get caught as outdated", (3, 4, 4))

    tdf = pd.DataFrame({"page": list(trust), "trust": list(trust.values())})
    tdf["genre"] = tdf.page.map(page_genre)
    order = tdf.groupby("genre").trust.median().sort_values().index

    @figure("03_method", "trust_by_genre", "Page trust is learned: first-person hype pages end up least trusted",
            f"eda/improve/trust.py (EM, {ROUNDS} rounds, genre prior pseudo-count {GENRE_PSEUDO}, page pseudo-count "
            f"{PSEUDO}); all practice labels as anchors", (4, 4, 4))
    def _(fig, t):
        ax = fig.add_subplot()
        for i, gname in enumerate(order):
            v = tdf[tdf.genre == gname].trust.values
            jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35
            ax.scatter(v, i + jitter, s=26, color=t["s"][0], alpha=0.55, edgecolor="none")
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=t["ink"], lw=2.5)
            ax.text(1.005, i, f"median {np.median(v):.2f}  · {len(v)} pages", va="center", fontsize=9,
                    color=t["ink2"], transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order], color=t["ink2"])
        ax.set_xlim(0.3, 1.0)
        ax.set_title("Learned page trust by page type")
        ax.set_xlabel("page trust (share of the page's mentions that agree with the staircase)")
        ax.grid(axis="y", visible=False)

    status_names = ["holds", "ahead_of_time", "unbacked", "outdated"]
    lab_x = L.R.set_index("id")["label"].reindex(X.index)
    worst = X["fs__worst"].map(dict(enumerate(status_names))).fillna("no fact matched")
    ct = pd.crosstab(worst, lab_x, normalize="index").reindex(columns=["true", "false", "unsure"]).fillna(0)
    cn = worst.value_counts()
    ct = ct.reindex([s for s in status_names + ["no fact matched"] if s in ct.index])

    @figure("03_method", "status_vs_label", "Each fact gets a named status; 'outdated' rows are mostly false",
            "out-of-fold timeline statuses (fold_group, eda/improve/failure.py) vs practice labels", (4, 5, 5))
    def _(fig, t):
        ax = fig.add_subplot()
        left = np.zeros(len(ct))
        y = np.arange(len(ct))
        for k in ["true", "false", "unsure"]:
            ax.barh(y, ct[k], left=left, color=t["s"][LABEL_SLOT[k]], height=0.6, label=k,
                    edgecolor=t["surface"], linewidth=2)
            for yi, (l, v) in enumerate(zip(left, ct[k])):
                if v >= 0.06:
                    ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=9,
                            color="#0b0b0b" if k == "unsure" and t["surface"] == "#fcfcfb" else "#ffffff")
            left += ct[k].values
        ax.set_yticks(y, [f"{s.replace('_', ' ')}  (n={cn.get(s, 0)})" for s in ct.index], color=t["ink2"])
        ax.set_xlim(0, 1)
        ax.set_title("Label mix by the row's worst fact status")
        ax.set_xlabel("share of rows")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(visible=False)

    stale = ((X["fs__any_outdated"] == 1) & (X["tl1__n_pages_against_max"] >= 1) & (X["tl1__support_margin_min"] < 0))
    single = X["pair_n_values_max"] == 1
    H = pd.DataFrame({
        "rule": ["outdated value, backed against → P(false) ≥ 0.8", "single-value pair → P(false) ≤ 0.05"],
        "fired": [int(stale.sum()), int(single.sum())],
        "agree": [float((lab_x[stale] == "false").mean()), float((lab_x[single] != "false").mean())],
    })

    @figure("03_method", "hard_rule_hits", "Two hard rules fire on 1 row in 5 and are right almost every time",
            "out-of-fold features (fold_group); rules in eda/07_improve.py:hard_rules", (3, 5, 4), size=(10, 3.8))
    def _(fig, t):
        ax = fig.add_subplot()
        y = np.arange(len(H))
        bars = ax.barh(y, H.fired, color=t["s"][0], height=0.5, edgecolor=t["surface"], linewidth=2)
        ax.set_xlim(0, H.fired.max() * 1.6)
        for b, (_, r) in zip(bars, H.iterrows()):
            ax.text(b.get_width() + 5, b.get_y() + b.get_height() / 2,
                    f"{r.fired} rows · rule agrees with the label {r.agree:.0%}", va="center", fontsize=10,
                    color=t["ink2"])
        ax.set_yticks(y, H.rule, color=t["ink2"])
        ax.set_title("Hard rules on top of the model")
        ax.set_xlabel(f"rows the rule fires on (of {len(X):,} cross-validated rows)")
        ax.grid(axis="y", visible=False)

    from improve.base import lgbm
    cols = L.cols_for("L6_judge", X)
    y = L.Y.reindex(X.index).to_numpy()
    m = lgbm().fit(X[cols], y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).nlargest(15).sort_values()
    p = m.get_params()
    ptxt = (f"num_leaves={p['num_leaves']} · n_estimators={p['n_estimators']} · learning_rate={p['learning_rate']} · "
            f"min_child_samples={p['min_child_samples']} · subsample={p['subsample']} · colsample={p['colsample_bytree']}"
            " · class weights: none")
    fam = lambda c: ("timeline" if c.startswith(("tl0__", "tl1__")) else "status" if c.startswith(("fs__", "tv__"))
                     else "date-aware KB" if c.startswith(("rel_", "days_since", "n_gold")) else "text / support")
    fams = ["timeline", "status", "date-aware KB", "text / support"]

    @figure("03_method", "lgbm_importance", "The timeline features carry the model; words barely register",
            "LightGBM (L6 feature set) fit on all cross-validated rows; share of total split gain", (4, 5, 5),
            size=(11, 6.2))
    def _(fig, t):
        ax = fig.add_subplot()
        colors = [t["s"][fams.index(fam(c))] for c in imp.index]
        ax.barh(range(len(imp)), imp.values, color=colors, height=0.65, edgecolor=t["surface"], linewidth=2)
        ax.set_yticks(range(len(imp)), [c.replace("__", " · ") for c in imp.index], fontsize=9, color=t["ink2"])
        for i, v in enumerate(imp.values):
            ax.text(v + imp.max() * 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=t["ink2"])
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=t["s"][i], label=f) for i, f in enumerate(fams)], loc="lower right", fontsize=9)
        ax.set_title("What the model listens to (top 15 features)")
        ax.set_xlabel("share of split gain")
        ax.grid(axis="y", visible=False)
        fig.text(0.01, 0.955, ptxt, fontsize=8.5, color=t["muted"])

    from improve.judge_nb import EvidenceJudge
    J = EvidenceJudge().fit(items, y == 0)
    lr = []
    for c, tab in J.tables.items():
        for k, v in tab.items():
            n_k = int((items[c].astype(str) == k).sum())
            if n_k >= 15:
                lr.append((f"{c} = {k}", v, n_k))
    LR = pd.DataFrame(lr, columns=["item", "log_lr", "n"])
    LR = pd.concat([LR.nlargest(8, "log_lr"), LR.nsmallest(6, "log_lr")]).drop_duplicates().sort_values("log_lr")
    temp = D["ladder"].query("variant=='L6_judge' and cv=='fold_group'").temperature.iloc[0]

    @figure("03_method", "evidence_ratios", "The evidence-sum judge: each clue moves the odds of 'false' by a fixed amount",
            f"eda/improve/judge_nb.py (Beta-smoothed, alpha=1) on out-of-fold items; temperature T={temp}",
            (4, 4, 4), size=(11, 6))
    def _(fig, t):
        ax = fig.add_subplot()
        colors = [t["s"][1] if v > 0 else t["s"][0] for v in LR.log_lr]
        ax.barh(range(len(LR)), LR.log_lr, color=colors, height=0.65, edgecolor=t["surface"], linewidth=2)
        ax.axvline(0, color=t["axis"], lw=1.2)
        lab_names = {"worst = 3.0": "worst status = outdated", "worst = 0.0": "worst status = holds",
                     "worst = 2.0": "worst status = unbacked", "worst = 1.0": "worst status = ahead of time",
                     "worst = -1.0": "no fact matched", "tv = F": "sentence state = false",
                     "tv = T": "sentence state = true", "tv = gap": "sentence state = evidence gap",
                     "joke = 1": "joke tail", "trust = 0": "page trust ≤ 0.6", "trust = 1": "page trust 0.6–0.8",
                     "trust = 2": "page trust 0.8–0.9", "trust = 3": "page trust > 0.9"}
        names = [lab_names.get(s, s.replace("archetype = ", "type: ").replace("_", " ")) for s in LR.item]
        ax.set_yticks(range(len(LR)), [f"{s}  (n={n})" for s, n in zip(names, LR.n)], fontsize=9, color=t["ink2"])
        for i, v in enumerate(LR.log_lr):
            ax.text(v + (0.05 if v > 0 else -0.05), i, f"×{np.exp(v):.2g}", va="center",
                    ha="left" if v > 0 else "right", fontsize=9, color=t["ink2"])
        ax.set_title("Likelihood ratio of 'false' per clue (log scale)")
        ax.set_xlabel("log likelihood ratio  ·  orange pushes towards false, blue towards true")
        ax.grid(axis="y", visible=False)

    @figure("03_method", "parameter_card", "The whole labelling method on one card: stages and every parameter",
            "eda/improve/*.py, eda/07_improve.py (variant L6), eda/out/improve_labels.csv",
            (2, 5, 5), size=(12, 4.4))
    def _(fig, t):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_xlim(0, 12)
        ax.set_ylim(1.9, 6.6)
        ax.text(0.3, 6.15, "How a row gets its label (L6)", fontsize=16, fontweight="bold", color=t["ink"])
        stages = [
            ("Facts", "subject · attribute · value\nper sentence\n(gold forms in research;\nextractor at run time)"),
            ("Date-aware KB", "value in force on the\nrow's date from training\nlabels; relation per fact"),
            ("Staircase", f"sticky path per pair\nANCHOR {ANCHOR} · SWITCH {SWITCH}\nTAU {TAU:g} d · new stair\nneeds ≥ 2 pages"),
            ("Page trust", f"EM, {ROUNDS} rounds\ngenre prior (pseudo {GENRE_PSEUDO:g})\npage pseudo-count {PSEUDO:g}\nweights the votes"),
            ("Statuses", "holds · ahead of time\nunbacked · outdated\n+ misread risk\n+ 3-valued sentence"),
            ("LightGBM", f"{p['num_leaves']} leaves · {p['n_estimators']} trees\nlr {p['learning_rate']} · no class weights\n+ hard rules\n(stale ≥ 0.8, single ≤ 0.05)"),
            ("Evidence sum", f"Beta-smoothed LRs\nper clue, logistic\ncalibrator + T = {temp}\nunsure band (tuned τ)"),
        ]
        wbox, gap = 1.5, 0.17
        for i, (h, body) in enumerate(stages):
            x = 0.3 + i * (wbox + gap)
            ax.add_patch(FancyBboxPatch((x, 3.75), wbox, 1.75, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=t["neutral"], edgecolor=t["axis"], linewidth=1))
            ax.text(x + 0.1, 5.3, h, fontsize=11.5, fontweight="bold", color=t["s"][0], va="top")
            ax.text(x + 0.08, 4.9, body, fontsize=8.6, color=t["ink2"], va="top", linespacing=1.4)
            if i < len(stages) - 1:
                ax.annotate("", (x + wbox + gap, 4.62), (x + wbox, 4.62),
                            arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.5))
        lad = D["ladder"]
        g = lad[lad.cv == "fold_group"].set_index("variant")
        ax.text(0.3, 3.2, "Cross-validated, pages held out (fold_group):", fontsize=11, color=t["ink"],
                fontweight="bold")
        ax.text(0.3, 2.75, f"base LightGBM  macro-F1 {g.loc['L0_base', 'macro_f1_tf']:.3f} · acc "
                f"{g.loc['L0_base', 'acc']:.1%} · ECE {g.loc['L0_base', 'ece']:.3f}", fontsize=11, color=t["ink2"])
        ax.text(0.3, 2.3, f"this method (L6)  macro-F1 {g.loc['L6_judge', 'macro_f1_tf']:.3f} · acc "
                f"{g.loc['L6_judge', 'acc']:.1%} · ECE {g.loc['L6_judge', 'ece']:.3f} · retrain "
                f"{g.loc['L7_runtime_retrain', 'fit_s']:.2f} s on 2 threads", fontsize=11, color=t["s"][0],
                fontweight="bold")


# ------------------------------------------------------------------ 04 results
NICE = {"L0_base": "base", "L1_timeline": "+ staircase", "L2_trust": "+ page trust", "L3_failure": "+ statuses",
        "L4_rules": "+ hard rules", "L5_threevalued": "+ 3-valued unsure", "L6_judge": "+ evidence sum",
        "L8_cfc": "+ liquid-net feature"}


def figs_results(D):
    variants = [v for v in NICE if v in D["oof"]["fold_group"]]
    rows = []
    for sch in ["fold_group", "fold_row5"]:
        for v in variants:
            if v not in D["oof"][sch]:
                continue
            f, lo, hi, g, glo, ghi = page_boot(D, sch, v)
            rows.append(dict(scheme=sch, variant=v, f1=f, lo=lo, hi=hi, gain=g, glo=glo, ghi=ghi))
    B = pd.DataFrame(rows)
    B.to_csv(SHOTS / "04_results" / "ladder_bootstrap.csv", index=False) if (SHOTS / "04_results").exists() else None

    @figure("04_results", "ladder_ci", "Every added piece holds up: macro-F1 0.735 → 0.80 with pages held out",
            "eda/out/improve_oof.pkl; 95% CI from 1,000 page-level bootstrap resamples", (4, 5, 5), size=(11, 5.6))
    def _(fig, t):
        axes = fig.subplots(1, 2, sharey=True)
        for ax, sch, ttl in zip(axes, ["fold_group", "fold_row5"],
                                ["pages held out (fold_group)", "check.sh-style rows held out (fold_row5)"]):
            b = B[B.scheme == sch].reset_index(drop=True)
            x = np.arange(len(b))
            col = [t["muted"] if v == "L0_base" else t["s"][0] for v in b.variant]
            ax.errorbar(x, b.f1, yerr=[b.f1 - b.lo, b.hi - b.f1], fmt="none", ecolor=t["axis"], elinewidth=2,
                        capsize=4, zorder=2)
            ax.scatter(x, b.f1, s=70, color=col, edgecolor=t["surface"], linewidth=2, zorder=3)
            for xi, v in zip(x, b.f1):
                ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=t["ink2"])
            ax.set_xticks(x, [NICE[v] for v in b.variant], rotation=35, ha="right", fontsize=9, color=t["ink2"])
            ax.set_title(ttl, fontsize=11, fontweight="normal", color=t["ink2"])
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("macro-F1 (true / false)")
        fig.suptitle("Improvement ladder: each step adds one piece", x=0.01, ha="left", fontsize=14,
                     fontweight="bold", color=t["ink"])

    g = B[B.scheme == "fold_group"].iloc[1:].reset_index(drop=True)

    @figure("04_results", "gain_vs_base", "The gain is real: every confidence interval clears zero across 248 pages",
            "paired page bootstrap vs base LightGBM (fold_group, 1,000 resamples)", (4, 5, 5))
    def _(fig, t):
        ax = fig.add_subplot()
        y = np.arange(len(g))
        ax.errorbar(g.gain, y, xerr=[g.gain - g.glo, g.ghi - g.gain], fmt="none", ecolor=t["axis"], elinewidth=2,
                    capsize=4)
        ax.scatter(g.gain, y, s=70, color=t["s"][0], edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.axvline(0, color=t["ink2"], lw=1.2, ls="--")
        for yi, (v, lo, hi) in enumerate(zip(g.gain, g.glo, g.ghi)):
            ax.text(hi + 0.004, yi, f"+{v:.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=9, color=t["ink2"])
        ax.set_yticks(y, [NICE[v] for v in g.variant], color=t["ink2"])
        ax.set_xlim(min(-0.01, g.glo.min() - 0.01), g.ghi.max() + 0.06)
        ax.set_title("Macro-F1 gain over the base model")
        ax.set_xlabel("gain in macro-F1 (95% CI)")
        ax.grid(axis="y", visible=False)

    o = D["oof"]["fold_group"]
    a0 = pd.DataFrame({"y": o["L0_base"]["y"], "p0": o["L0_base"]["pred"]}, index=o["L0_base"]["ids"])
    a6 = pd.DataFrame({"p6": o["L6_judge"]["pred"]}, index=o["L6_judge"]["ids"])
    A = a0.join(a6)
    A["arch"] = D["ann"]["archetype"].reindex(A.index)
    per = A.groupby("arch").apply(lambda d: pd.Series({"n": len(d), "acc0": (d.y == d.p0).mean(),
                                                       "acc6": (d.y == d.p6).mean()}))
    per = per[per.n >= 20].sort_values("acc6")

    @figure("04_results", "per_archetype", "The biggest jumps are on outdated values and contradictions",
            "out-of-fold predictions by row archetype (eda/out/practice_annotated.csv), archetypes with ≥ 20 rows",
            (3, 4, 4), size=(11, 6.2))
    def _(fig, t):
        ax = fig.add_subplot()
        y = np.arange(len(per))
        for yi, (a, b) in enumerate(zip(per.acc0, per.acc6)):
            ax.plot([a, b], [yi, yi], color=t["axis"], lw=2, zorder=1)
        ax.scatter(per.acc0, y, s=60, color=t["muted"], label="base", zorder=2, edgecolor=t["surface"])
        ax.scatter(per.acc6, y, s=60, color=t["s"][0], label="this method (L6)", zorder=3, edgecolor=t["surface"])
        ax.set_yticks(y, [f"{a.replace('_', ' ')}  (n={int(n)})" for a, n in zip(per.index, per.n)], fontsize=9,
                      color=t["ink2"])
        ax.set_title("Accuracy per row type: base → this method")
        ax.set_xlabel("accuracy")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)

    names = ["false", "true", "unsure"]

    @figure("04_results", "confusion_base_vs_L6", "Fewer missed lies: more false rows caught, few true rows lost",
            "out-of-fold predictions (fold_group, 1,593 rows)", (3, 4, 4), size=(11, 4.8))
    def _(fig, t):
        axes = fig.subplots(1, 2)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", SEQ if t["surface"] == "#fcfcfb" else SEQ[::-1])
        for ax, col, ttl in zip(axes, ["p0", "p6"], ["base", "this method (L6)"]):
            cm = confusion_matrix(A.y, A[col], labels=[0, 1, 2])
            norm = cm / cm.sum(1, keepdims=True)
            ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    dark_cell = (norm[i, j] > 0.55) == (t["surface"] == "#fcfcfb")
                    ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                            color="#ffffff" if dark_cell else t["ink"])
            ax.set_xticks(range(3), names, color=t["ink2"])
            ax.set_yticks(range(3), names, color=t["ink2"])
            ax.set_xlabel("predicted")
            ax.set_ylabel("actual")
            ax.set_title(ttl, fontsize=11, fontweight="normal", color=t["ink2"])
            ax.grid(False)
        fig.suptitle("Confusion matrices (row-normalised)", x=0.01, ha="left", fontsize=14, fontweight="bold",
                     color=t["ink"])


# ------------------------------------------------------------------ 05 calibration
def figs_calibration(D):
    o = D["oof"]["fold_group"]

    def rel(v):
        P, pred, y = np.asarray(o[v]["proba"]), np.asarray(o[v]["pred"]), np.asarray(o[v]["y"])
        m = pred != 2
        conf = P[np.arange(len(P)), pred][m]
        ok = (pred == y)[m]
        bins = np.linspace(0.5, 1.0, 6)
        k = np.clip(np.digitize(conf, bins) - 1, 0, 4)
        return pd.DataFrame({"conf": conf, "ok": ok, "k": k}).groupby("k").agg(
            conf=("conf", "mean"), acc=("ok", "mean"), n=("ok", "size"))

    r0, r6 = rel("L0_base"), rel("L6_judge")
    e0 = D["ladder"].query("variant=='L0_base' and cv=='fold_group'").ece.iloc[0]
    e6 = D["ladder"].query("variant=='L6_judge' and cv=='fold_group'").ece.iloc[0]

    @figure("05_calibration", "reliability", "Confidence means what it says: when we say 90%, we're right ≥ 90%",
            "out-of-fold probabilities (fold_group); 5 equal-width confidence bins", (4, 5, 5), size=(7.5, 6.5))
    def _(fig, t):
        ax = fig.add_subplot()
        ax.plot([0.5, 1], [0.5, 1], color=t["axis"], lw=1.5, ls="--", label="perfect calibration")
        ax.plot(r0.conf, r0.acc, "-o", color=t["muted"], ms=8, label=f"base (ECE {e0:.3f})")
        ax.plot(r6.conf, r6.acc, "-o", color=t["s"][0], ms=8, label=f"this method (ECE {e6:.3f})",
                markeredgecolor=t["surface"], markeredgewidth=1.5)
        ax.set_xlim(0.48, 1.01)
        ax.set_ylim(0.48, 1.01)
        ax.set_xlabel("stated confidence")
        ax.set_ylabel("share actually correct")
        ax.set_title("Reliability diagram")
        ax.legend(loc="upper left")

    cb = pd.read_csv(OUT / "handoff" / "calibration_bins.csv").query("scheme=='fold_group' and variant=='L6_judge'")

    @figure("05_calibration", "confidence_bins", "Our 0.85–0.95 band is right 94% of the time; 'all true' is right 83–85%",
            "eda/out/handoff/calibration_bins.csv; baselines: all-true on 1,897 rows (83.3%) and on check.sh's 380 "
            "hidden rows (85.3%)", (4, 5, 5))
    def _(fig, t):
        ax = fig.add_subplot()
        x = np.arange(len(cb))
        bars = ax.bar(x, cb.share_correct, color=t["s"][0], width=0.6, edgecolor=t["surface"], linewidth=2)
        for b, v, n, c in zip(bars, cb.share_correct, cb.rows, cb.mean_conf):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}\n{n} rows · says {c:.2f}", ha="center",
                    fontsize=9, color=t["ink2"])
        for val, txt, dy, va in [(0.833, "all true, every row: 83.3%", -0.006, "top"),
                                 (0.853, "all true, check.sh's hidden rows: 85.3%", 0.006, "bottom")]:
            ax.axhline(val, color=t["ink2"], lw=1.2, ls="--")
            ax.text(-0.45, val + dy, txt, ha="left", va=va, fontsize=8.5, color=t["ink2"],
                    bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1))
        ax.set_xticks(x, cb.band, color=t["ink2"])
        ax.set_ylim(0.5, 1.08)
        ax.set_xlabel("confidence band the model states")
        ax.set_ylabel("share correct")
        ax.set_title("How often each confidence band is right")
        ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------ 06 neural
def figs_neural(D):
    S = D["sweep"].copy()
    S["cell"] = S.variant.str.extract(r"n15_(\w+?)_h")[0]
    S["hidden"] = S.variant.str.extract(r"_h(\d+)_")[0].astype(int)
    S["dt"] = S.variant.str.extract(r"_(raw|log|none)$")[0]
    agg = S.groupby(["cell", "hidden", "dt"]).macro_f1_tf.agg(["mean", "std"]).reset_index()
    ref = 0.742

    @figure("06_neural", "liquid_sweep", "Liquid and recurrent nets trail the tree model by 4–6 points on true/false",
            "eda/out/neuro_sweep_seeds.csv (27 configs × 3 seeds × 5 folds, RTX 5050); reference LightGBM on the same folds",
            (3, 4, 4), size=(11, 5.4))
    def _(fig, t):
        axes = fig.subplots(1, 3, sharey=True)
        for ax, dt in zip(axes, ["log", "raw", "none"]):
            for ci, cell in enumerate(["gru", "ltc", "cfc"]):
                a = agg[(agg.cell == cell) & (agg.dt == dt)].sort_values("hidden")
                xs = np.arange(len(a)) + (ci - 1) * 0.18
                ax.errorbar(xs, a["mean"], yerr=a["std"], fmt="o", color=t["s"][ci], ms=7, capsize=3, lw=1.5,
                            label=cell.upper())
            ax.axhline(ref, color=t["ink2"], ls="--", lw=1.2)
            ax.set_xticks(range(3), ["16", "32", "64"], color=t["ink2"])
            ax.set_xlabel("hidden units")
            ax.set_title(f"time gaps: {dt}", fontsize=11, fontweight="normal", color=t["ink2"])
            ax.grid(axis="x", visible=False)
        axes[0].text(0.05, ref + 0.003, "LightGBM 0.742", fontsize=9, color=t["ink2"], transform=axes[0].get_yaxis_transform())
        axes[0].set_ylabel("macro-F1 (mean ± std over seeds)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.99))
        fig.suptitle("Recurrent sweep: GRU ≥ LTC ≥ CfC; log time gaps help every cell", x=0.01, ha="left",
                     fontsize=14, fontweight="bold", color=t["ink"])

    N = D["neuro"]
    pick = {"n13_lgbm_reference": "LightGBM", "n13_cfc_unsure_head": "CfC unsure head",
            "n13_snn_unsure_head": "spiking unsure head", "n13_stack_lgbm_cfc_snn": "stack: tree + CfC + spiking",
            "n14_lgbm70_bigru30": "tree 70% + char-BiGRU 30%", "n14_char_bigru": "char-BiGRU alone",
            "n14_char_cnn": "char-CNN alone"}
    P = N[N.variant.isin(pick)].drop_duplicates("variant").set_index("variant")

    @figure("06_neural", "unsure_vs_accuracy", "Neural heads find unsure rows (up to 25%) but pay in accuracy",
            "eda/out/improve_neuro.csv (items 13–14, same folds)", (4, 4, 4))
    def _(fig, t):
        ax = fig.add_subplot()
        for i, (v, name) in enumerate(pick.items()):
            if v not in P.index:
                continue
            r = P.loc[v]
            col = t["muted"] if "LightGBM" == name else t["s"][min(i, 7)]
            ax.scatter(r.acc, r.unsure_recall, s=90, color=col, edgecolor=t["surface"], linewidth=2, zorder=3)
            ax.annotate(name, (r.acc, r.unsure_recall), xytext=(6, 6), textcoords="offset points", fontsize=9,
                        color=t["ink2"])
        ax.set_xlabel("accuracy")
        ax.set_ylabel("unsure recall")
        ax.set_title("The unsure / accuracy trade-off")

    C = N.dropna(subset=["predict_s_cpu2"]).copy()
    C = C[C.variant.str.match(r"n1[2-4]_|n15_(gru|cfc|ltc)_h(16|64)_log")].drop_duplicates("variant")
    C["ms"] = C.predict_s_cpu2 * 1000
    C = C.sort_values("ms")

    @figure("06_neural", "cpu_time", "Every model fits the 2-CPU run: the slowest needs 12 s for 20,000 rows",
            "eda/out/improve_neuro.csv (predict time on 2 threads, scaled to 20,000 rows)", (2, 4, 4), size=(10, 6))
    def _(fig, t):
        ax = fig.add_subplot()
        ax.barh(range(len(C)), C.ms, color=t["s"][0], height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.set_xscale("log")
        ax.set_yticks(range(len(C)), [v.replace("n1", "").replace("_", " ") for v in C.variant], fontsize=8.5,
                      color=t["ink2"])
        for i, v in enumerate(C.ms):
            ax.text(v * 1.08, i, f"{v / 1000:.2f} s", va="center", fontsize=8.5, color=t["ink2"])
        ax.set_xlabel("ms to predict 20,000 rows on 2 CPU threads (log scale)")
        ax.set_title("Inference cost per model")
        ax.grid(axis="y", visible=False)


# ------------------------------------------------------------------ 07 extraction
def figs_extraction(D):
    E = D["extr"]
    E = E[E.fold.astype(str).str.isdigit()] if E.fold.dtype == object else E
    k = E.groupby("kind")[["gold", "extracted", "correct"]].sum()
    k["f1"] = 2 * k.correct / (k.gold + k.extracted)
    allf = k.loc["ALL", "f1"] if "ALL" in k.index else np.nan
    k = k.drop(index="ALL", errors="ignore").sort_values("f1")

    @figure("07_extraction", "f1_per_kind", "A simple extractor reads 72% of facts; names and yes/no are the weak spots",
            "eda/out/extraction.csv (out of fold vs the gold forms)", (3, 4, 4))
    def _(fig, t):
        ax = fig.add_subplot()
        bars = ax.barh(k.index, k.f1, color=t["s"][0], height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.axvline(allf, color=t["ink2"], lw=1.2, ls="--")
        ax.text(allf, len(k) - 0.4, f" overall F1 {allf:.2f}", color=t["ink2"], fontsize=9)
        for b, (_, r) in zip(bars, k.iterrows()):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{r.f1:.2f}  (gold n={int(r.gold)})",
                    va="center", fontsize=9, color=t["ink2"])
        ax.set_xlim(0, 1.1)
        ax.set_title("Fact extraction F1 by value kind")
        ax.set_xlabel("F1 against the gold facts")
        ax.grid(axis="y", visible=False)

    cf = D["conf"].nlargest(10, "count").sort_values("count")

    @figure("07_extraction", "confusable_attributes", "Attribute mix-ups are near-synonyms: refund window vs free trial",
            "eda/out/confusable_attributes.csv (attribute step, out of fold)", (3, 4, 3))
    def _(fig, t):
        ax = fig.add_subplot()
        ax.barh(range(len(cf)), cf["count"], color=t["s"][1], height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.set_yticks(range(len(cf)), [f"{a} → {b}" for a, b in zip(cf.gold_attribute, cf.predicted_attribute)],
                      fontsize=9, color=t["ink2"])
        for i, v in enumerate(cf["count"]):
            ax.text(v + 0.1, i, str(v), va="center", fontsize=9, color=t["ink2"])
        ax.set_title("Most confused attributes (gold → predicted)")
        ax.set_xlabel("facts")
        ax.grid(axis="y", visible=False)


# ------------------------------------------------------------------ 08 self-improvement
def figs_selfimprove(D):
    trials = ROOT / "eda" / "selfimprove" / "trials.csv"
    d = SHOTS / "08_selfimprove"
    d.mkdir(parents=True, exist_ok=True)
    if not trials.exists():
        (d / "README.md").write_text(
            "# 08 · self-improvement\n\nNo `eda/selfimprove/trials.csv` yet. When the 60-minute loop has run, "
            "re-run `.venv/Scripts/python eda/09_figures.py` and the trial chart appears here.\n", encoding="utf-8")
        print("  08_selfimprove: placeholder (no trials yet)")
        return
    T = pd.read_csv(trials)
    gcol = next((c for c in ["gain", "delta", "mean_gain", "delta_macro_f1"] if c in T.columns), None)
    kcol = next((c for c in ["kept", "accepted", "accept", "decision"] if c in T.columns), None)
    ncol = next((c for c in ["change", "knob", "trial", "name", "variant"] if c in T.columns), T.columns[0])
    if gcol is None:
        print("  08_selfimprove: trials.csv has no gain column; skipped")
        return
    T = T[T.get("knob", pd.Series("x", index=T.index)).astype(str) != "-"].reset_index(drop=True)
    if "knob" in T.columns and "value" in T.columns:
        T["label"] = T["knob"].astype(str) + " = " + T["value"].astype(str)
        if "folds_better" in T.columns:
            T["label"] += "  (" + T["folds_better"].astype(str) + "/5 folds better)"
        ncol = "label"
    kept = T[kcol].astype(str).str.lower().isin(["true", "1", "kept", "accept", "accepted", "yes"]) if kcol else \
        pd.Series(False, index=T.index)
    (d / "README.md").unlink(missing_ok=True)

    @figure("08_selfimprove", "trials", "The loop keeps a change only when it wins on most folds",
            "eda/selfimprove/trials.csv", (3, 4, 4), size=(11, max(4, 0.32 * len(T) + 1.5)))
    def _(fig, t):
        ax = fig.add_subplot()
        cols = [t["s"][0] if k else t["muted"] for k in kept]
        ax.barh(range(len(T)), T[gcol], color=cols, height=0.6, edgecolor=t["surface"], linewidth=2)
        ax.axvline(0, color=t["axis"], lw=1.2)
        ax.set_yticks(range(len(T)), [str(s)[:50] for s in T[ncol]], fontsize=8.5, color=t["ink2"])
        ax.invert_yaxis()
        ax.set_xlabel("change in mean macro-F1 vs the current best")
        ax.set_title("Self-improvement trials (blue = kept)")
        ax.grid(axis="y", visible=False)


# ------------------------------------------------------------------ index
def write_index():
    I = pd.DataFrame(INDEX)
    cats = {"01_data": "Data", "02_rules": "Rules found in the data", "03_method": "The method and its parameters",
            "04_results": "Results", "05_calibration": "Calibration", "06_neural": "Neural track",
            "07_extraction": "Extraction", "08_selfimprove": "Self-improvement loop"}
    lines = ["# Figure library (menu for the pitch)", "",
             "Generated by `.venv/Scripts/python eda/09_figures.py`. Every figure exists as `.png` and `.svg`, light and "
             "`-dark`. **This is a menu to choose from: nothing here is wired into `pitch/PITCH.md`.** The project "
             "isn't finished, so numbers will move; re-run the script after each new result.", "",
             "**Insight score** = geometric mean of three 1–5 sub-scores: surprise (would a judge expect it?), clarity "
             "(readable in 5 seconds?) and relevance (does it support the story: the date decides truth, and our "
             "confidence is honest?).", ""]
    for c, title in cats.items():
        s = I[I.cat == c]
        if s.empty:
            continue
        lines += [f"## {title}", "", "| Figure | Takeaway | Score | Surprise · Clarity · Relevance | Data |",
                  "|---|---|---|---|---|"]
        for r in s.itertuples():
            lines.append(f"| [`{r.path}`]({r.path}) | {r.takeaway} | **{r.score}** | {r.surprise} · {r.clarity} · "
                         f"{r.relevance} | {r.source} |")
        lines.append("")
    top = I.sort_values(["score", "relevance"], ascending=False)
    why = {
        "02_rules/fake_strike.png": "opens the story in one picture: same words, different date, different truth",
        "03_method/parameter_card.png": "the architecture slide with real parameters and the before/after numbers",
        "05_calibration/reliability.png": "proves confidence is honest, which the jury scores directly",
        "04_results/gain_vs_base.png": "shows the improvement is statistically real, not noise",
        "03_method/staircase_gemini_response.png": "the core idea (staircase + trusted pages) on real data",
        "04_results/ladder_ci.png": "what each piece adds, with uncertainty",
        "02_rules/rule_false_rates.png": "the rules we found in the data, with evidence",
        "05_calibration/confidence_bins.png": "confidence bands vs both all-true baselines",
        "03_method/lgbm_importance.png": "the model relies on time, not on words",
        "03_method/status_vs_label.png": "named statuses make every verdict explainable",
        "03_method/evidence_ratios.png": "explainability: each clue's weight",
        "06_neural/unsure_vs_accuracy.png": "honest neuromorphic slide: where liquid/spiking nets help",
    }
    picks = [p for p in why if p in set(I.path)][:8]
    lines += ["## Suggested top 8 for the pitch", "",
              "Ordered for a 5-minute story (problem → idea → method → proof → honesty). Swap freely.", ""]
    for i, p in enumerate(picks, 1):
        r = I[I.path == p].iloc[0]
        lines.append(f"{i}. [`{p}`]({p}) (score {r.score}): {why[p]}.")
    lines += ["", f"_All figures by score:_ " + ", ".join(f"`{p}` {s}" for p, s in zip(top.path, top.score)), ""]
    (SHOTS / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    fresh = "--fresh" in sys.argv
    SHOTS.mkdir(parents=True, exist_ok=True)
    (SHOTS / "04_results").mkdir(parents=True, exist_ok=True)
    D = load()
    print("data"); figs_data(D)
    print("rules"); figs_rules(D)
    print("oof features (cached after the first run)")
    X, items = oof_feature_table(fresh)
    print("method"); figs_method(D, X, items)
    print("results"); figs_results(D)
    print("calibration"); figs_calibration(D)
    print("neural"); figs_neural(D)
    print("extraction"); figs_extraction(D)
    print("self-improvement"); figs_selfimprove(D)
    write_index()
    print(f"{len(INDEX)} figures -> {SHOTS}")


if __name__ == "__main__":
    main()
