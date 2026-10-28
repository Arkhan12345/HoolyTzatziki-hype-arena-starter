"""Pitch figures as SVG, light + dark (P5). Standard library only: python3 pitch/figs/make_figs.py

Reads data/practice.csv, eda/out/bakeoff.csv and eda/out/ablation.csv; writes pitch/figs/<name>.svg and <name>-dark.svg.
Colours: validated categorical slots 1-3 (dataviz skill palette), one fixed meaning in every chart:
slot 1 blue = true, slot 2 orange = false, slot 3 aqua = unsure. Text never wears a series colour.
"""
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  true="#2a78d6", false="#eb6834", unsure="#1baf7a", other="#c3c2b7", loss="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 true="#3987e5", false="#d95926", unsure="#199e70", other="#52514e", loss="#e66767"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
W, H = 960, 480


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, fill, size=15, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family=\'{FONT}\'>{esc(s)}</text>')


def hbar(x0, x1, yc, h, fill, tip):
    """Horizontal bar from baseline x0 to data end x1: 4px rounded data end, square at the baseline."""
    r = min(4, abs(x1 - x0) / 2)
    y0, y1 = yc - h / 2, yc + h / 2
    if x1 >= x0:
        d = f"M{x0},{y0} H{x1 - r} Q{x1},{y0} {x1},{y0 + r} V{y1 - r} Q{x1},{y1} {x1 - r},{y1} H{x0} Z"
    else:
        d = f"M{x0},{y0} H{x1 + r} Q{x1},{y0} {x1},{y0 + r} V{y1 - r} Q{x1},{y1} {x1 + r},{y1} H{x0} Z"
    return f'<path d="{d}" fill="{fill}"><title>{esc(tip)}</title></path>'


def svg(body, t, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="{esc(label)}"><rect width="{W}" height="{H}" fill="{t["surface"]}"/>{"".join(body)}</svg>\n')


def source(t, s):
    return text(24, H - 16, f"Source: {s}", t["muted"], 12)


# 1 · label mix -------------------------------------------------------------------------------------------------------
def label_mix(t):
    rows = list(csv.DictReader(open(ROOT / "data/practice.csv", encoding="utf-8")))
    n = len(rows)
    counts = [(lab, sum(r["label"] == lab for r in rows)) for lab in ("true", "false", "unsure")]
    x0, x1 = 140, 800
    scale = (x1 - x0) / max(c for _, c in counts)
    body = [text(24, 40, f"{n:,} labelled practice rows", t["ink"], 20, weight=600),
            text(24, 64, "Most sentences are true: a model that says 'true' to everything is right 83% of the time",
                 t["ink2"], 14)]
    for i, (lab, c) in enumerate(counts):
        yc = 150 + i * 90
        body.append(text(x0 - 16, yc + 5, lab, t["ink"], 18, "end", 600))
        body.append(hbar(x0, x0 + c * scale, yc, 24, t[lab], f"{lab}: {c:,} rows"))
        body.append(text(x0 + c * scale + 12, yc + 6, f"{c:,}  ({c / n:.0%})", t["ink"], 17))
    body.append(f'<line x1="{x0}" y1="110" x2="{x0}" y2="370" stroke="{t["axis"]}" stroke-width="1"/>')
    body.append(source(t, "data/practice.csv"))
    return svg(body, t, "Label mix of the practice rows: 1,581 true, 277 false, 39 unsure")


# 2 · the fake strike: Hooly Gemini median_response_time ---------------------------------------------------------------
def fake_strike(t):
    pts = []
    for r in csv.DictReader(open(ROOT / "data/practice.csv", encoding="utf-8")):
        for f in json.loads(r["form"] or "[]"):
            if f["subject"] == "Hooly Gemini" and f["attribute"] == "median_response_time" and "op" not in f["value"]:
                pts.append((date.fromisoformat(r["date"]), float(f["value"]["value"]), r["label"], r["id"], r["text"]))
    d0, d1 = date(2025, 10, 1), date(2026, 11, 1)
    L, R, T, B = 90, 920, 110, 400
    X = lambda d: L + (d - d0).days / (d1 - d0).days * (R - L)
    Y = lambda v: B - v / 2.5 * (B - T)
    change = date(2026, 3, 24)
    body = [text(24, 40, "Same sentence, different date, different truth", t["ink"], 20, weight=600),
            text(24, 64, "Hooly Gemini's median reply time: every practice sentence that states it, by date",
                 t["ink2"], 14)]
    for v in (0, 0.5, 1, 1.5, 2, 2.5):
        body.append(f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>')
        body.append(text(L - 10, Y(v) + 5, f"{v:g} s", t["muted"], 13, "end"))
    for m in (date(2025, 10, 1), date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1), date(2026, 10, 1)):
        body.append(text(X(m), B + 24, m.strftime("%b %Y"), t["muted"], 13, "middle"))
    # value in force: 2.1 s until Gemini 2.0 (2026-03-24), then 1.4 s
    first = date(2025, 10, 10)
    body.append(f'<polyline points="{X(first)},{Y(2.1)} {X(change)},{Y(2.1)} {X(change)},{Y(1.4)} {R},{Y(1.4)}" '
                f'fill="none" stroke="{t["ink2"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    body.append(text(X(change) + 8, Y(1.4) - 52, "24 Mar 2026: Gemini 2.0", t["ink2"], 13))
    body.append(text(X(change) + 8, Y(1.4) - 36, "median drops to 1.4 s", t["ink2"], 13))
    for d, v, lab, rid, txt in sorted(pts, key=lambda p: p[2] == "false"):
        body.append(f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="6" fill="{t[lab]}" stroke="{t["surface"]}" '
                    f'stroke-width="2"><title>{esc(f"{d} · {lab} · {rid}: {txt}")}</title></circle>')
    # the two rows with the identical sentence
    for d, lab in ((date(2025, 11, 14), "true"), (date(2026, 8, 20), "false")):
        body.append(f'<line x1="{X(d):.1f}" y1="{Y(2.1) - 9:.1f}" x2="{X(d):.1f}" y2="{Y(2.1) - 22:.1f}" '
                    f'stroke="{t["ink2"]}" stroke-width="1"/>')
        body.append(text(X(d), Y(2.1) - 28, f"{d:%d %b %Y}: {lab}", t["ink"], 14, "middle", 600))
    body.append(text(X(date(2026, 8, 20)), Y(2.1) - 46, "“Hooly Gemini replies in 2.1 s at the median.”",
                     t["ink"], 14, "middle"))
    # legend
    lx = 300
    for i, (key, name) in enumerate((("true", "true row"), ("false", "false row"))):
        body.append(f'<circle cx="{lx + i * 120}" cy="88" r="6" fill="{t[key]}"/>')
        body.append(text(lx + i * 120 + 12, 93, name, t["ink2"], 13))
    body.append(f'<line x1="{lx + 240}" y1="88" x2="{lx + 262}" y2="88" stroke="{t["ink2"]}" stroke-width="2"/>')
    body.append(text(lx + 270, 93, "value in force", t["ink2"], 13))
    body.append(source(t, "data/practice.csv (rows with an exact value; p-1459 and p-1474 are the same sentence)"))
    return svg(body, t, "Timeline of Hooly Gemini's median reply time: 2.1 s until 24 March 2026, then 1.4 s. "
                        "The same sentence is true on 14 Nov 2025 and false on 20 Aug 2026.")


# 3 · leaderboard -----------------------------------------------------------------------------------------------------
MODELS = [("lightgbm_unweighted", "LightGBM (ships)"), ("extratrees", "ExtraTrees"),
          ("cfc_liquid_timeline", "CfC liquid net"), ("spiking_mlp_lif", "Spiking MLP (LIF)"),
          ("tfidf_logreg_text_only", "Words only (TF-IDF)"), ("majority_all_true", "Everything true")]


def leaderboard(t):
    f1 = {r["model"]: float(r["macro_f1_tf"]) for r in csv.DictReader(open(ROOT / "eda/out/bakeoff.csv"))
          if r["cv"] == "group"}
    x0, x1, vmax = 230, 860, 0.8
    body = [text(24, 40, "Trees win, and the date-aware features are what they win with", t["ink"], 20, weight=600),
            text(24, 64, "Macro-F1 over true/false, 5-fold cross-validation grouped by page", t["ink2"], 14)]
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        x = x0 + v / vmax * (x1 - x0)
        body.append(f'<line x1="{x}" y1="92" x2="{x}" y2="420" stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>')
        body.append(text(x, 440, f"{v:.1f}", t["muted"], 13, "middle"))
    for i, (key, name) in enumerate(MODELS):
        yc, v = 116 + i * 58, f1[key]
        fill = t["true"] if i == 0 else t["other"]
        body.append(text(x0 - 14, yc + 5, name, t["ink"], 16, "end", 600 if i == 0 else 400))
        body.append(hbar(x0, x0 + v / vmax * (x1 - x0), yc, 24, fill, f"{name}: {v:.3f}"))
        body.append(text(x0 + v / vmax * (x1 - x0) + 10, yc + 5, f"{v:.2f}", t["ink"], 15))
    body.append(source(t, "eda/out/bakeoff.csv (cv = group)"))
    return svg(body, t, "Macro-F1 by model: LightGBM 0.76, ExtraTrees 0.72, CfC 0.70, spiking 0.66, "
                        "words only 0.56, everything true 0.45")


# 4 · ablation ---------------------------------------------------------------------------------------------------------
FAMILIES = [("temporal_kb", "Date-aware knowledge base"), ("page_genre", "Page type"), ("value_shape", "Value shape"),
            ("date", "Dates in the text"), ("support", "Label-free support"), ("wording", "Wording cues")]


def ablation(t):
    d = {r["minus_family"]: float(r["delta_vs_all"]) for r in csv.DictReader(open(ROOT / "eda/out/ablation.csv"))}
    lo, hi = -0.08, 0.02
    x0, x1 = 300, 900
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    body = [text(24, 40, "Take the timeline away and the model loses most", t["ink"], 20, weight=600),
            text(24, 64, "Change in LightGBM macro-F1 when one feature family is removed (group CV)", t["ink2"], 14)]
    for v in (-0.08, -0.06, -0.04, -0.02, 0, 0.02):
        body.append(f'<line x1="{X(v)}" y1="92" x2="{X(v)}" y2="420" stroke="{t["axis"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        body.append(text(X(v), 440, f"{v:+.2f}" if v else "0", t["muted"], 13, "middle"))
    for i, (key, name) in enumerate(FAMILIES):
        yc, v = 116 + i * 58, d[key]
        body.append(text(24, yc + 5, name, t["ink"], 16, weight=600 if i == 0 else 400))
        if abs(v) > 1e-9:
            body.append(hbar(X(0), X(v), yc, 24, t["loss"] if v < 0 else t["other"], f"without {name}: {v:+.3f}"))
        lab = f"{v:+.3f}" if abs(v) > 1e-9 else "0.000"
        body.append(text(X(v) - 10 if v < 0 else X(v) + 10, yc + 5, lab, t["ink"], 15, "end" if v < 0 else "start"))
    body.append(source(t, "eda/out/ablation.csv"))
    return svg(body, t, "Ablation: without the date-aware knowledge base LightGBM loses 0.075 macro-F1; "
                        "every other family changes it by less than 0.015")


if __name__ == "__main__":
    for name, fn in (("label_mix", label_mix), ("fake_strike", fake_strike), ("leaderboard", leaderboard),
                     ("ablation", ablation)):
        for mode, t in THEMES.items():
            p = OUT / (f"{name}.svg" if mode == "light" else f"{name}-dark.svg")
            p.write_text(fn(t), encoding="utf-8")
            print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.1f} KB")
