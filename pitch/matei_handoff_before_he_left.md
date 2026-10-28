# Matei (P5) handoff: pitch, slides, video, Q&A

Written 28 Oct 2026, 15:20. For whoever takes over the pitch (X by default). Everything below is in `pitch/`.

## State

| File | What | State |
|---|---|---|
| `STORY.md` | the story + every number with its source file and owner | done; refresh when numbers change |
| `PITCH.md` | 10 slides for `/pitch`, with speaker notes (`> notes:`) | v1 done; live keys checked |
| `QA.md` | 16 jury questions, short sourced answers | done |
| `VIDEO_SCRIPT.md` | 3-min shot list for 29 Oct 09:00 + rehearsal checklist | done; not rehearsed yet |
| `figs/*.svg` | label mix, Gemini fake-strike timeline, leaderboard, ablation (light + `-dark`) | done; not on the slides yet |
| `figs/make_figs.py` | regenerates the figures: `python3 pitch/figs/make_figs.py` (stdlib only) | done |
| `export/` | PDF/PPTX backup of the deck | **not done** (do at 08:30 freeze) |

PRs: #11 (`matei/pitch`) holds the deck, story, Q&A, video script and figures. This handoff is on top of it.

## Open items, in order

1. **Merge #11** (and this PR). `/pitch` shows a placeholder until `pitch/PITCH.md` is on main.
2. **Figures on the slides.** Waiting on P4: `frontend/scripts/sync-assets.mjs` must copy `pitch/figs/` → `public/figs/`
   (Request in `STATUS.md`). Then add `![](figs/label_mix.svg)` to slide 2, `fake_strike` to slide 3, and `leaderboard` +
   `ablation` to slide 9. Pick the `-dark.svg` files if the deck renders dark.
3. **Classifier numbers.** When Arckan writes the `model` key in `pitch/metrics.json`, slide 7 fills
   `{{offline.data.model.accuracy|pct}}` / `{{offline.data.model.macro_f1|pct}}` by itself. Add the numbers to `STORY.md` (rows L1, L2)
   and update `QA.md` answers 7–8 if they differ from the research CV (0.76 group, 0.62 time-forward).
4. **Rehearse the video** on practice rows before 09:00 (checklist at the bottom of `VIDEO_SCRIPT.md`).
5. **08:30 freeze:** export the deck to `pitch/export/` (PDF), plus a screenshot of `/architecture` for the backup.
6. **09:00–10:00:** record the video on `sample/sample.csv`. **17:00:** present from `/pitch`.

## Gotchas (learned the hard way)

- **Never write a bare `{{model.*}}`.** `/pitch` reads the latest run's `/metrics` first; after a demo run with no classifier
  that is the all-true baseline's score. Use `{{offline.data.model.*}}`.
- **No markdown tables in `PITCH.md`.** SlideDeck uses `react-markdown` without `remark-gfm`; tables render as raw pipes.
- **Slide format:** `## Slide N — title`, body, `> notes:` lines. Text before the first slide is ignored.
- **Claim precision is 0.17% / 0.67% / 1.9%** wrong at 10 / 25 / 50% pages hidden (JOURNAL 15:10). The older 0.5% in
  `Matei_pitch+presentation.md` is out of date.
- **The prompt names `copy-pitch.mjs`;** it is now `frontend/scripts/sync-assets.mjs`.
- **No Python 3.12 venv on Matei's laptop;** that's why the figures are hand-written SVG instead of matplotlib.
- **The sample has no labels:** never claim accuracy on it in the video, only predictions and explanations.
- **Numbers rule:** a number not in `STORY.md` is not said on stage. Model scores M1–M10 are X's research CV, not the shipped model.

## One line to remember

**"The date decides truth."** The same Gemini sentence is true on 14 Nov 2025 and false on 20 Aug 2026 (p-1459, p-1474).
