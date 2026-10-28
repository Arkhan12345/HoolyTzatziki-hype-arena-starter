# X · P1 · Research + Architecture (v2, from 28 Oct ~14:00) — Team Tzatziki · Hype or Truth

> **How to use:** open Claude Code in your clone of https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter, `git checkout main && git pull`, and paste this WHOLE file as the first message.
> This replaces `team/X_architecture+pitch.md` for X from now on. **The pitch, slides, video and Q&A now belong to Matei (P5)**; his prompt is [`Matei_pitch+presentation.md`](Matei_pitch+presentation.md). X keeps research, architecture, integration and the final PR.

## FILES THIS PROMPT CREATES / MODIFIES
You (Claude) may READ every file in the repo. You may only CREATE or MODIFY these:

```text
docs/  (ARCHITECTURE, REQUIREMENTS, EDA, EDA_v2, MODELS, RULES, DECISIONS (new), RESEARCH_LOG (new))
eda/   (research scripts + eda/out/ + eda/splits.csv; shared with Arckan: he reads, X writes)
README.md (team section only; never the starter text above the divider) · CLAUDE.md · STATUS.md (structure + own row)
app/contracts.py · team/README.md · X_research+architecture.md · Matei_pitch+presentation.md
pitch/JOURNAL.md (append only) · pitch/metrics.json (only the "research" key)
CONDITIONAL (only after Arckan's "neuro handoff" Request): train/neuro/ · train/neuro/LOG.md · models/neuro/
PR description to QA-DNA on 29 Oct
```
Anything else → a Request line in STATUS.md to its owner. Never touch `contracts/`, `data/`, `sample/`, upstream `scripts/check.sh`, `input.py`, `check_output.py`.

## ENVIRONMENT (follow DEPENDENCIES.md, always)
- Read `CLAUDE.md`, `DEPENDENCIES.md`, `team/README.md`, `STATUS.md` before anything else.
- Runtime image: `python:3.12.16-slim-trixie`, exact pins in `requirements.txt` + `constraints.txt` (owner P2 Ioan). **Never edit those**; need a package → Request to P2.
- Shared venv (classic EDA, tests): `python -m venv .venv` → `.venv/Scripts/python -m pip install -r requirements-dev.txt -r eda/requirements-eda.txt` (`eda/requirements-eda.txt` already pulls in `requirements.txt` + `constraints.txt`). Rebuild the EDA outputs: `.venv/Scripts/python eda/01_parse.py` → `02_rules.py` → `03_recommend.py` → `05_rules_doc.py` → `06_splits.py`.
- Neuro venv (CfC / LTC / SNN bake-off only): `.venv-neuro` from `eda/requirements-eda-neuro.txt` (torch 2.14.1+cpu, ncps 1.0.1, snntorch 1.0.0). Never in the image unless a neuro model wins AND P2's `WITH_NEURO=1` build arg exists.
- Exact `==` pins only. A new research package = add it pinned to `eda/requirements-eda*.txt` + one line in `DEPENDENCIES.md` (dev-only section) + one line in `pitch/JOURNAL.md`.

## STATE OF THE PROJECT (read, don't redo)
- **Backend (P2 Ioan) on main (PRs #4–#6):** Dockerfile, `app/run.py` (fallback table, watchdog), temporal KB, claim generator v2 + validator (1,000 claims, 100% validator-clean, each with its source sentences), measured claim precision + recalibrated confidences (`scripts/claim_holdout.py`, expected wrong claims 2.3% → 0.5%), demo API v1.1, metrics (`pitch/metrics.json` → `runtime`), `scripts/check.sh` green in Docker (689 MB image, 19k rows in ~4 s). PR #7 (claims from gold vs from extraction) is open. P3 plugs in through `app.extract.extract(rows, lexicon)` and `app.classify.classify(rows, facts, kb)`; KB features in `app/kb/features.py`; EM hook `kb.timelines()` / `kb.set_reliability()`; classifier output can feed the claims via `KB.add_classified` (`TZ_TRUST_CLASSIFIED`).
- **Model (P3 Arckan):** extractor + classifier not merged yet → labels are still the all-true baseline.
- **Frontend (P4 Vald):** scaffold on main; redesign on `vald/frontend-slop-2`. `/pitch` renders `pitch/PITCH.md` (`## Slide N` per slide, `> notes:`, live numbers via `{{model.accuracy|pct}}`).
- **Research (X, done):** `docs/EDA_v2.md`, `docs/RULES.md` (every rule in one file), `eda/out/practice_annotated.csv` (per-row parameters + archetype + option A–F), `eda/splits.csv` (shared folds), bake-off in `eda/out/bakeoff.csv`.
  Key findings:
  - the date-aware KB carries the model (−0.075 macro-F1 without it);
  - stale value → 84% false; a pair with a single value in the corpus → never false;
  - unsure = joke/hype tail or an unverifiable one-off value;
  - LightGBM unweighted 0.76 group-CV vs CfC 0.70, SNN 0.66, text-only 0.56, all-true 0.455; time-forward 0.62 (pessimistic).
- **Chosen architecture so far:** option **A** (temporal KB → unweighted LightGBM + tuned unsure threshold + hard rules "stale → false", "single-value pair → never false"); next **B** (page trust via EM) and **D** (CfC "still current" as one feature); **E** pitch only; **F** gated on QA DNA's answer.
- **Open question to QA DNA:** https://github.com/QA-DNA/hype-arena-starter/issues/2. Their only answer so far: "you cannot call anything external in your final solution". Our follow-up (10:58, from Alexx1612) asks whether a fine-tuned small encoder (MiniLM / DistilBERT / BERT-base) shipped inside the image counts as external, and whether weights only are enough or the base model + training script must ship too. Unanswered. Until answered: no pretrained models.
- **Already done by X (don't redo):** Matei's row in `team/README.md` + `STATUS.md`; the "@P3 from P1" research hand-off Request; the dev-only EDA section in `DEPENDENCIES.md` (Request to P2 to confirm); `data/RULES.md` → `docs/RULES.md` and `data/splits.csv` → `eda/splits.csv` (data/ is upstream-owned).

## TEAM (5 people, parallel)
P1 X — research + architecture + integration + final PR · P2 Ioan — backend/devops · P3 Arckan — extractor/classifier/models · P4 Vald — frontend · **P5 Matei — pitch, slides, video, Q&A, presentation of findings (new)**.
Ownership table: `team/README.md`. Same teamwork rules as before (below).

## MISSION
Turn the research into decisions the model owner can ship, keep the architecture coherent across 5 agents, and make sure what we pitch is what we run.

## TASKS, IN ORDER
1. **Matei up and running (first 20 min).** Make sure Arkhan12345 added him as a collaborator and that he has pasted `Matei_pitch+presentation.md`. Ask Vald (Request) to change the comment in `frontend/src/pages/Pitch.tsx` from "P1's file" to "P5's file". Give Matei the research numbers he asks for and answer his questions within one pull cycle.
2. **Follow up with Arckan.** The "@P3 from P1" hand-off Request is posted (folds in `eda/splits.csv`, features in `eda/features.py` / `docs/EDA_v2.md` §4, the two hard rules, unweighted LightGBM + unsure threshold). Check each pull whether he answered; when he merges `app/classify`, re-score it on `eda/splits.csv` (`fold_group`, then `holdout` once) and post the numbers. You never edit `app/`.
3. **Research backlog (in order, each ≤ 1 h, each ends with a row in `eda/out/bakeoff.csv` and a line in `docs/RESEARCH_LOG.md`):**
   a. Unsure threshold + calibration (isotonic vs Platt) on OOF probabilities → ECE and macro-F1 incl. unsure.
   b. Truth discovery (option B): Dawid-Skene / TruthFinder page reliability as features; ablation with/without.
   c. Label-free KB only (what the run has for unlabelled rows): rebuild features without gold timeline for the test fold → realistic score.
   d. Extraction-noise test: drop/perturb 15% of gold facts → how much does macro-F1 fall (sets the extraction target for Arckan).
   e. CfC "still current" probability as one LightGBM feature (option D) in `.venv-neuro`; ships only if group CV improves and inference < 2 min.
   f. Claim-side research for P2: which fact pairs are stable after 2026-09 (safe for `fact` claims dated 29 Oct).
   Re-run `eda/05_rules_doc.py` after any rule change so `docs/RULES.md` stays the single rule file.
4. **Architecture.** Keep `docs/ARCHITECTURE.md` + README options table in sync with what is merged; log every choice in `docs/DECISIONS.md` (`date | decision | evidence | owner`). Additive interface changes only in `app/contracts.py`.
5. **Integration checks at T2 (22:00) and T3 (08:30):** fetch + merge main, `.venv/Scripts/python -m pytest -q`, `MINUTES=10 sh scripts/check.sh` (WSL), record results in STATUS Milestones + JOURNAL; failures → precise Request to the owner.
6. **Watch issue #2** every pull (`gh api repos/QA-DNA/hype-arena-starter/issues/2/comments --jq '.[] | "\(.user.login): \(.body)"'`; or a Claude Code `Monitor` that polls it every 2 min). When answered: log it in `docs/DECISIONS.md`, Request P2 to update `DEPENDENCIES.md` rule 3, and tell Arckan whether option F is allowed.
7. **29 Oct:** 09:00 `git pull upstream main` (sample), final check.sh; ≤ 10:30 open the PR to QA-DNA titled "Tzatziki" (`gh pr create --repo QA-DNA/hype-arena-starter --head Arkhan12345:main --title "Tzatziki"`), body: what it is, how to start the frontend, Dockerfile at root. 10:45 announce freeze; after 11:00 nobody pushes.

## TEAMWORK RULES (unchanged, non-negotiable)
1. Parallel, never blocked: code against interfaces/stubs; never wait, never ask anyone to wait.
2. Start of every task: `git fetch origin && git merge origin/main && git log --oneline -15 origin/main && cat STATUS.md`.
3. Branches `X/<topic>`; commit small (`P1-research: …` / `P1-architecture: …`) and push after every commit.
4. Merge to main via PR to OUR repo as soon as green, at least every 1–2 h (`gh pr create --repo Arkhan12345/HoolyTzatziki-hype-arena-starter --base main --fill` → `gh pr merge --merge --delete-branch`).
5. Edit only your folders; everything else is a STATUS Request.
6. One `pitch/JOURNAL.md` line per meaningful merge: `HH:MM | P1 | what | why / number`.
7. Update your STATUS row hourly. 8. No secrets, no files > 20 MB. 9. No LLM and no network at run time.

## DEFINITION OF DONE
Arckan ships a classifier that uses the research (beats 0.83 accuracy and 0.455 macro-F1 on hidden practice rows); every decision has evidence in `docs/DECISIONS.md`; `docs/RULES.md` current; check.sh green at T3; PR opened before 10:30 on 29 Oct.
