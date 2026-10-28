# Matei · P5 · Pitch + Presentation + Findings (new member) — Team Tzatziki · Hype or Truth

> **How to use:** clone https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter (ask Arkhan12345 to add you as a collaborator), run `claude` in the repo, and paste this WHOLE file as the first message.
> Read first, in this order: `CLAUDE.md`, `team/README.md`, `STATUS.md`, `README.md` (team section), `docs/EDA_v2.md`, `docs/RULES.md`, `docs/ARCHITECTURE.md`, `docs/MODELS.md`, `pitch/JOURNAL.md`, `pitch/metrics.json`, `frontend/src/pages/Pitch.tsx`.

## FILES THIS PROMPT CREATES / MODIFIES
You (Claude) may READ every file in the repo. You may only CREATE or MODIFY these:

```text
pitch/PITCH.md          the deck (rendered live by the frontend at /pitch)
pitch/QA.md             jury questions + short answers
pitch/VIDEO_SCRIPT.md   the ≤ 3 min video (recorded 29 Oct after 09:00)
pitch/STORY.md          the narrative + numbers sheet (single source for every number you say)
pitch/figs/             charts/diagrams for the slides (PNG/SVG, < 1 MB each)
pitch/export/           offline backup of the deck (PDF/PPTX)
pitch/JOURNAL.md        append only · STATUS.md: your own row + your Requests only
```
Anything else (docs, eda, app, frontend, models) → a Request line in STATUS.md to its owner. Never touch `contracts/`, `data/`, `sample/`.

## ENVIRONMENT (follow DEPENDENCIES.md)
- Read `DEPENDENCIES.md`. No new runtime packages, ever. For charts use the shared venv:
  `python -m venv .venv` → `.venv/Scripts/python -m pip install -r eda/requirements-eda.txt` (pulls in the image's exact pins + matplotlib).
- Frontend preview of your deck: `cd frontend && npm ci && npm run dev` → `/pitch` (Node 22+; `scripts/copy-pitch.mjs` copies `pitch/PITCH.md` into the app). Do not edit `frontend/`; ask Vald (P4) via STATUS.
- Live numbers in slides: write `{{key.path}}` (optionally `|pct`) in PITCH.md; the values come from `pitch/metrics.json` through `/metrics` (syntax: `frontend/src/pages/Pitch.tsx`). Today metrics.json only has `runtime.*` (e.g. `runtime.practice_run.total_s`, `runtime.claims_by_type.compare`, `runtime.claim_holdout.*`); `model.*` appears when Arckan's classifier lands. Use only keys that exist, check `/pitch` renders them, and never hard-code a number metrics.json will hold.

## SHARED BRIEF (short)
~~~text
CHALLENGE (QA DNA "Hype or Truth", NEXXT AI Hackathon, 28–29 Oct 2026)
A fictional company, Hooly: a year of its pages where prices, people and dates change and some pages lie.
Our system (1) labels ~18,000 rows true / false / unsure with a confidence 0.5–1, and (2) writes exactly
1,000 new claims, every one true. An LLM may help BUILD it but may not BE it. Offline Docker, 2 CPUs,
6 GiB, 30 min. Frontend: architecture diagram + live run on the 20 sample rows (29 Oct 09:00) + video.
PR "Tzatziki" to QA-DNA before 11:00 on 29 Oct (X opens it). Final pitch 17:00 if selected.

TEAM (5 Claude Code sessions in parallel)
P1 X — research + architecture + integration + final PR   P2 Ioan — Docker, run, KB, claims, API
P3 Arckan — extractor, classifier, models                  P4 Vald — frontend (/, /run, /architecture, /metrics, /pitch)
P5 Matei — YOU: pitch, slides, video, Q&A, presenting the data findings and the chosen architecture

RULES: branches matei/<topic>; small commits "P5-pitch: …", push after every commit; merge via PR to OUR repo
when green (gh pr create --repo Arkhan12345/HoolyTzatziki-hype-arena-starter --base main --fill, then
gh pr merge --merge --delete-branch), at least every 1–2 h. Start every task with
git fetch origin && git merge origin/main && git log --oneline -15 origin/main && cat STATUS.md.
Never block or wait for anyone. One JOURNAL line per merge: HH:MM | P5 | what | why / number.
~~~

## WHAT IS ALREADY TRUE (your raw material; cite the file for every number)
- **Data** (`docs/EDA_v2.md`, `docs/RULES.md`):
  - 1,897 labelled rows: 83% true, 15% false, 2% unsure; 291 pages over 159 dates; 2,402 facts; 733 with only a pronoun.
  - **The date decides truth.** A fact repeating an older value after it changed is false 84% of the time (the "fake strike": "Gemini replies in 2.1 s" is true in Nov 2025, false in Aug 2026). A pair with one value in the whole corpus is never false.
  - **Lying pages:** first-person hype pages are 40% false; joke tails ("…and the curve is vertical") make rows unsure.
  - **Generated universe:** a fixed pool of Silicon Valley names, repeated templates, and false numbers made by multiplying the true one (×2, ×0.5, ×10).
- **Models** (`eda/out/bakeoff.csv`, macro-F1 true/false, grouped CV):
  - LightGBM 0.76, ExtraTrees 0.72, CfC liquid 0.70, spiking LIF 0.66, text-only 0.56, all-true 0.455.
  - Without the date-aware knowledge base, LightGBM loses 0.075.
  - Liquid and spiking nets catch more unsure rows (28–40% vs ~0) but calibrate worse.
- **Chosen architecture** (`README.md` options, `docs/ARCHITECTURE.md`): text → extractor → temporal knowledge base → LightGBM + two hard rules → calibrated label; KB → claim generator → validator → exactly 1,000 claims. Option A ships; B (page trust) and D (liquid feature) are being tested; E (spiking) is our energy/neuromorphic story; F (pretrained encoder) waits on QA DNA issue #2.
- **Runtime + claims** (`pitch/metrics.json` → `runtime`, `pitch/JOURNAL.md`): image 689 MB of 3 GiB, ~4 s for 19k rows, 1,000/1,000 claims validator-clean and each traceable to its source sentences, measured claim precision (generate from half the pages, judge on all) put expected wrong claims at 0.5%, check.sh green in Docker.
- **Classifier:** not merged yet, so live labels are still the all-true baseline. Write the results slide with `{{model.*}}` keys and a fallback sentence until the numbers land.
- **Open with QA DNA:** issue #2 (is a fine-tuned small encoder allowed?): unanswered. In Q&A say we use no pretrained model until they confirm.

## MISSION
A 5-minute pitch and a 3-minute video that a jury believes: every claim backed by a number from our files, the architecture shown as it really runs, and the data story ("the date decides truth") told so simply that a non-ML judge repeats it.

## TASKS, IN ORDER
1. **[first 30 min] `pitch/STORY.md`:** one-paragraph story + a numbers table (number | meaning | source file | owner). Every number in slides, video and Q&A comes from this table. Mark numbers that will change (model scores) as `{{…}}` keys.
2. **[≤ 1 h] `pitch/PITCH.md` v1** in the frontend's format (`## Slide N — title`, body, `> notes:` speaker notes, ~45 s per slide):
   1 Title + one-liner · 2 The problem (a universe where facts change and pages lie) · 3 What the data taught us (fake strike, single-value pairs, lying pages, generator) · 4 Architecture (Vald's diagram; extractor → temporal KB → LightGBM + rules → generator → validator) · 5 Why no LLM inside and why that's better (explainable, offline, 2 CPUs, auditable) · 6 Live demo (sample.csv) · 7 Results (`{{model.*}}` once they exist, claims 1,000/1,000 + measured claim precision, runtime) · 8 Confidence is part of the answer (calibration) · 9 Model leaderboard: classic vs liquid vs spiking (honest: trees win; neuro wins on unsure, energy) · 10 Team + next steps.
   Merge it to main early so `/pitch` stops showing the placeholder.
3. **[≤ 1 h] Figures in `pitch/figs/`:** label mix, fake-strike timeline example, leaderboard bar (from `eda/out/bakeoff.csv`), ablation bar (`eda/out/ablation.csv`), architecture (screenshot from `/architecture`). Reuse `eda/out/figs/*.png` where good enough. Same colours across all charts.
4. **`pitch/QA.md`:** 15 likely jury questions with 2–3-line answers + the source file. Include: why not an LLM; is a pretrained encoder allowed (issue #2 status); how lying pages are handled; how confidence is calibrated; what if an entity is unseen; overfitting with 1,900 labels (grouped CV, holdout pages, rules over memorisation); runtime and scaling to 20k+; failure modes; why neuromorphic at all.
5. **`pitch/VIDEO_SCRIPT.md` (≤ 3 min):** diagram → upload `sample/sample.csv` on `/run` → labels + confidence → claims → metrics. Shot list with timings and what to click. Rehearse on practice rows before 09:00 on 29 Oct.
6. **[every 2 h] Refresh:** pull, read new JOURNAL lines and metrics.json, update STORY.md numbers and slides; JOURNAL line per merge. Ask X (P1) for research numbers, Arckan (P3) for model numbers, Ioan (P2) for runtime, Vald (P4) for rendering issues, all via STATUS Requests.
7. **[T3 08:30, 29 Oct] Freeze the deck**; export a PDF/PPTX backup to `pitch/export/`. 09:00–10:00 record the video with the team on the 20 sample rows. 17:00: present from `/pitch`.

## RECOMMENDED CLAUDE CODE SETUP
Model Opus, effort medium. Use `dataviz` (charts), `anthropic-skills:pptx` / `pdf` (offline export), and the claude.ai Slides artifact for a shareable link. Use plan mode before rewriting the deck. Run `/loop 60m` to refresh numbers and your STATUS row.

## DEFINITION OF DONE
PITCH.md renders at `/pitch` with live numbers and no placeholder; every number traceable in STORY.md; QA.md has 15 answers; the video script has been rehearsed; the PDF backup is exported. All of it merged to main before 10:30 on 29 Oct.
