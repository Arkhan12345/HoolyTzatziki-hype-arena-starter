# Team Tzatziki — 5 prompts, parallel work

**Challenge:** QA DNA "Hype or Truth" (NEXXT AI Hackathon, 28–29 Oct 2026). Label ~18,000 rows about the
fictional company Hooly as true / false / unsure with confidence, and generate exactly 1,000 new true claims.
An LLM may help build the system but may not be it.

**Fork:** https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter (upstream `QA-DNA/hype-arena-starter`)
**Hard deadline:** PR titled "Tzatziki" opened before **11:00 on 29 Oct**; no pushes after 11:00.

## Who gets which file

| # | Role | Member | File to paste into Claude Code | Owns |
|---|---|---|---|---|
| 1 | Research + Architecture + Integration | X | [`X_research+architecture.md`](../X_research+architecture.md) (v2; was [`X_architecture+pitch.md`](X_architecture+pitch.md)) | `docs/` `eda/` (research, shared with Arckan) `README.md` `STATUS.md` `CLAUDE.md` `app/contracts.py` |
| 2 | Backend + DevOps | Ioan | [`Ioan_backend+devops.md`](Ioan_backend+devops.md) | `Dockerfile` `app/run.py` `app/io` `app/kb` `app/generate` `app/validate` `app/api` `scripts/` (new) `tests/runtime` |
| 3 | Backend + AI (the model) | Arckan | [`Arckan_backend+ai.md`](Arckan_backend+ai.md) | `app/extract` `app/classify` `app/neuro` `models/` `eda/` `train/` `tests/model` |
| 4 | Fullstack + Frontend | Vald | [`Vald_fullstack+frontend.md`](Vald_fullstack+frontend.md) | `frontend/` |
| 5 | Pitch + Presentation + Findings (new) | Matei | [`Matei_pitch+presentation.md`](../Matei_pitch+presentation.md) | `pitch/PITCH.md` `pitch/QA.md` `pitch/VIDEO_SCRIPT.md` `pitch/STORY.md` `pitch/figs/` `pitch/export/` |

Shared reference docs (in `docs/` since the first commit):
[`docs/REQUIREMENTS.md`](../docs/REQUIREMENTS.md) · [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) · [`docs/EDA.md`](../docs/EDA.md) · [`docs/MODELS.md`](../docs/MODELS.md) · architecture options: [`README.md`](../README.md#proposed-architecture-options-for-review)

## How to start (every member, on your own laptop)

1. **Get access:** the fork owner (Arkhan12345) adds X, Ioan, Arckan and Vald as collaborators (`gh repo edit` or GitHub → Settings → Collaborators).
2. **Clone and add upstream:**
   ```sh
   git clone https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter.git
   cd hype-arena-starter
   git remote add upstream https://github.com/QA-DNA/hype-arena-starter.git
   ```
3. **Start your agent:** run `claude` in the repo and paste your WHOLE file from `team/` as the first message.
4. **Order:**
   - P1 pushes the skeleton, `contracts.py`, `CLAUDE.md` and `STATUS.md` within 45 min.
   - P2, P3 and P4 start immediately in their own folders. They never wait; they pull P1's push when it lands.

## Team rhythm (also in every prompt and in `CLAUDE.md`)

- **All work is on branches:** one branch per task, named `<name>/<topic>` (`X/contracts`, `ioan/dockerfile`, `arckan/extractor`, `vald/run-page`). Nobody commits directly to `main`.
- **Commits:** small, every working step and at least every 30 min, as `P#-role: what`. **Push the branch after every commit** so teammates can see the progress.
- **Merging:** as soon as a step is green (tests or build), merge it to `main` through a PR to **our** repo and merge it yourself (`gh pr create --repo Arkhan12345/HoolyTzatziki-hype-arena-starter --base main --fill`, then `gh pr merge --merge --delete-branch`). Do this at least every 1–2 h. X reviews afterwards and never blocks a merge.
- **Before each task:** `git fetch origin && git merge origin/main`, `git log origin/main`, `cat STATUS.md`.
- **Ownership:** only edit your own folders. Anything you need from someone else goes in a `STATUS.md` → Requests line.
- **Journal:** one line in `pitch/JOURNAL.md` per meaningful merge. This is the living pitch.
- **Status:** update your `STATUS.md` row every hour.

## Milestones

- [ ] **T0 (+45 min):** skeleton, contracts and stubs pushed; Dockerfile passes `scripts/check.sh`.
- [ ] **T1 (28 Oct, 16:00):** walking skeleton. Stub pipeline end to end; the frontend calls `/run`.
- [ ] **T2 (28 Oct, 22:00):** real extractor, KB and LightGBM beat the baseline; 1,000 valid claims; the frontend shows real data.
- [ ] **T3 (29 Oct, 08:30):** feature freeze; check.sh green; pitch has real numbers.
- [ ] **29 Oct, 09:00:** `git pull upstream main` (gets `sample/sample.csv`); live run; record the video.
- [ ] **29 Oct, ≤10:30:** PR "Tzatziki" opened. **11:00: frozen.**
- [ ] **29 Oct, 17:00:** final pitch from the `/pitch` page.
