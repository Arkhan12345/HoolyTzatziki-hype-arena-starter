# Vald · P4 · Fullstack + Frontend — Team Tzatziki · Hype or Truth

> **How to use:** open Claude Code in your clone of https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter and paste this WHOLE file as the first message (Shared Brief + your role prompt). Read `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/EDA.md`, `docs/MODELS.md` and `team/README.md` first.

## FILES THIS PROMPT CREATES / MODIFIES
You (Claude) may READ every file in the repo. You may only CREATE or MODIFY these:

```text
frontend/package.json · vite.config.ts · tsconfig*.json · tailwind/postcss config · components.json (shadcn)
frontend/src/main.tsx · App.tsx · router · src/api/client.ts · src/mocks/{run,architecture,metrics}.json
frontend/src/pages/{Run,Architecture,Metrics,Pitch}.tsx
frontend/src/components/ (Watermelon UI + Cult UI copies, Stamp, ConfidenceBar, FactChip, PipelineFlow, SlideDeck)
frontend/README.md · frontend/scripts/start.ps1 · frontend/scripts/start.sh
pitch/JOURNAL.md (append) · STATUS.md (own row + requests)
```
Anything else → a Request line in STATUS.md to its owner.

## RECOMMENDED CLAUDE CODE SETUP (skills, tools, harness) for this session
Install or enable these before pasting the prompt. Anything marked *(if available)* depends on your account or plugins, so skip it if it's missing; nothing below is required to start.

**Everyone (same baseline on all 4 laptops):**
- **Model:** Opus via `/model`. Effort **high** for design/rules work, **medium** for routine edits.
- **CLIs:** `gh` (logged in with `gh auth login`), `git`, Python 3.12, Docker Desktop. On Windows, run `scripts/check.sh` from Git Bash or WSL.
- **Shared harness:** X commits `.claude/settings.json`, so every session gets the same setup. It holds:
  - a **SessionStart hook** that runs `git fetch origin && git log --oneline -10 origin/main && git branch -r && cat STATUS.md`, so the agent sees teammates' commits every time it starts;
  - a **permissions allowlist** for `git pull/add/commit/push`, `python -m pytest`, `npm run *`, `docker build/run`, `sh scripts/check.sh` (fewer prompts, faster parallel work). Afterwards, run `/fewer-permission-prompts` to add the rest.
- **Context7 MCP** for up-to-date library docs: `claude mcp add context7 -- npx -y @upstash/context7-mcp`.
- **Built-ins to use:**
  - **Plan mode** (Shift+Tab) before any multi-file change.
  - **Explore subagent** to read contracts/ and teammates' code without flooding context.
  - **`run_in_background`** for long commands (training, docker build, check.sh).
  - **`/code-review`** on your own diff before pushing big changes.
  - **`/loop 60m`** to refresh your STATUS.md row and read new Requests.

**For Vald (fullstack + frontend):**
- **shadcn MCP server** to browse and add registry components, including **Watermelon UI** and **Cult UI**: `npx shadcn@latest mcp init --client claude`. Then add components by their registry URLs.
- **Playwright MCP** to screenshot and click through the UI and verify the demo flow: `claude mcp add playwright -- npx -y @playwright/mcp@latest`. Alternatively, use the **claude-in-chrome** skill *(if available)*.
- **`/run` skill** to start `npm run dev` + the API and check pages actually render.
- **frontend-design plugin** *(if available, from the official Claude Code plugins marketplace)* for layout and visual polish.
- **`dataviz` skill** *(if available)* before building the metrics and calibration charts.
- **Context7** for React Flow (@xyflow/react), Recharts, Tailwind, Vite.
- **Hook (your own `.claude/settings.local.json`):** a PostToolUse hook that runs `npm run build --prefix frontend` after edits in `frontend/src/**`, so main never gets a broken build.

## SHARED BRIEF (identical in all 4 files — read it fully before your role section)

~~~text
=== SHARED BRIEF — Team Tzatziki · "Hype or Truth" (QA DNA, NEXXT AI Hackathon, 28–29 Oct 2026) ===

THE CHALLENGE
Build a system that writes new claims about a fictional tech company (Hooly) whose prices, people and
dates change and whose pages sometimes lie. It labels each row true / false / unsure and says how sure
it is. THE RULE: an LLM may help us BUILD it, but may NOT BE it (no LLM inside the shipped system).
The source material is the whole world: never use the web or a pretrained model's knowledge as truth.

TWO JOBS, ONE SYSTEM
1. Label every unlabelled row (~18,000 of ~20,000): true | false | unsure, confidence 0.5–1.0
   (empty confidence for unsure). false = the row states at least one false value on its date.
2. Generate EXACTLY 1,000 new claims, every one true and new (not a copy of an input row), each a
   schema-valid entry of type fact | compare | change | at_event | relation, with confidence 0.5–1.0.

REPO: our fork https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter  (upstream: QA-DNA/hype-arena-starter)
Read first: README.md, contracts/README.md, contracts/claim-forms.md, contracts/value-forms.md,
contracts/entry.schema.json, contracts/claimed_value.schema.json, data/practice.csv, data/lexicon.json,
scripts/check.sh. Team docs (already in the repo): docs/REQUIREMENTS.md, docs/ARCHITECTURE.md, docs/EDA.md, team/README.md.

HARD RUNTIME CONTRACT (break one = zero score)
- One Docker image from ./Dockerfile, `docker build --platform linux/amd64 .`, ≤ 3 GiB unpacked.
- Dockerfile ends with `USER 10001`. Build has network; RUN HAS NONE.
- 2 CPUs, 6 GiB RAM, ≤256 procs/threads, read-only root; writable only /tmp (128 MB, counts in RAM)
  and /output (256 MB). No home dir: set HOME, XDG_CACHE_HOME, HF_HOME, MPLCONFIGDIR to /tmp.
- 30 minutes; SIGTERM at deadline, kill 10 s later; env ARENA_DEADLINE = deadline (unix seconds).
- Input (read-only): /universe/universe.csv (id,text,date,label — NO form column) + /universe/lexicon.json.
- Output: /output/output.csv ≤16 MiB, columns id,type,text,form,label,confidence. Write to a temp
  name in /output and os.replace() to output.csv when complete. Write a fallback table EARLY.
- The image MAY contain data/practice.csv (forms included) and our trained models.

STACK
Python 3.12 · pandas/csv · scikit-learn · lightgbm · jsonschema · joblib · FastAPI + uvicorn (demo API)
Frontend: React + Vite + TypeScript + Tailwind + shadcn/ui, components from Watermelon UI and Cult UI.
Neuro track (dev-time, ships only if it wins): torch-CPU + ncps (CfC/LTC) + snntorch + nir in
requirements-neuro.txt, Docker build arg WITH_NEURO=1. Full model plan: docs/MODELS.md.

MODEL (agreed, v2): temporal KB + truth discovery + a fairly-compared MODEL ZOO
 (classic: LogReg/SVM/KNN/NB/RF/ExtraTrees/LightGBM/MLP/stacks; liquid: CfC/LTC; spiking: SNN/LSM).
 The leaderboard winner ships; fallback = the LightGBM+LogReg stack. Options: README.md. Pipeline:
 text ─► extractor (lexicon aliases + value regexes + TF-IDF/LR attribute classifier + same-page
 coreference) ─► facts (subject, attribute, value) ─► temporal KB (practice gold + votes across the
 universe + per-page trust) ─► features ─► LightGBM ─► calibrated label+confidence
 KB ─► claim templates (claim-forms.md) ─► validator ─► exactly 1,000 claims ─► output.csv

TEAM — 4 people, 4 laptops, 4 Claude Code sessions, ALL WORKING IN PARALLEL
 P1 X — Architect + Pitch + Structure (skeleton, contracts.py, docs, STATUS, integration, pitch)
 P2 Ioan — Backend + DevOps (Dockerfile, run.py, I/O, KB, claim generator, validator, API)
 P3 Arckan — Backend + AI (extractor, classifier, calibration, EDA, training, optional neuromorphic)
 P4 Vald — Fullstack + Frontend (architecture diagram, live run on sample.csv, metrics, /pitch page)
You are ONE of these four agents. The other three exist, are working right now, and push to the same
repo. Acknowledge them: refer to them by role, read their commits, build on their work.

FOLDER OWNERSHIP (only the owner edits; others write a request in STATUS.md → "Requests")
 P1: docs/  pitch/PITCH.md  pitch/QA.md  README.md  STATUS.md (structure)  CLAUDE.md  app/contracts.py
 P2: Dockerfile  .dockerignore  requirements.txt  app/run.py  app/io/  app/kb/  app/generate/
     app/validate/  app/api/  scripts/ (new files only; never edit upstream check.sh/input.py/
     check_output.py)  tests/runtime/
 P3: app/extract/  app/classify/  app/neuro/  models/  eda/  train/  tests/model/  requirements-train.txt
     requirements-neuro.txt   (train/neuro/ + models/neuro/ move to X ONLY after Arckan's handoff Request)
 P4: frontend/
 Shared append-only: pitch/JOURNAL.md (everyone appends), STATUS.md (each edits ONLY own row/requests),
 pitch/metrics.json (P3 writes model metrics, P2 writes runtime metrics under its own key).
 Never touch: contracts/, data/, sample/ (upstream-owned).

SHARED INTERFACES (app/contracts.py, written by P1 in the first 45 min; stubs return valid shapes)
 Fact(row_id, subject, attribute, value: dict, span: str, conf: float)
 Label(id, label: "true"|"false"|"unsure", confidence: float|None)
 Claim(id, type, text, form: dict, confidence: float)
 extract(rows, lexicon) -> list[Fact]                      # P3, app/extract
 KB.build(facts, gold_rows) ; KB.lookup(subject, attr, date) -> (value, support, history)   # P2, app/kb
 classify(rows, facts, kb) -> list[Label]                  # P3, app/classify
 generate(kb, lexicon, input_texts, n=1000) -> list[Claim] # P2, app/generate
 validate_claim(claim, lexicon, input_texts) -> list[str]  # P2, app/validate (empty list = valid)
 pipeline: app/run.py main(universe_dir, output_dir, deadline) -> writes output.csv, returns timings
 API (P2): POST /run (multipart CSV or {"use":"sample"}) -> {labels, claims, facts, timings, stages}
           GET /architecture -> stage graph JSON ; GET /metrics -> pitch/metrics.json
 Changing an interface = P1 edits contracts.py, keeps the old signature working, announces it in
 STATUS.md and JOURNAL.md. Never break a teammate silently.

ORDER & TIMELINE (now ≈ 11:00, 28 Oct)
 T0  +45 min   P1 pushes skeleton + contracts.py + stubs + CLAUDE.md + STATUS.md (everyone else
               meanwhile reads the contracts/data locally and drafts in their own folder — no waiting).
               P2 pushes Dockerfile + dummy run.py passing scripts/check.sh.
 T1  16:00     WALKING SKELETON: stubs end-to-end → output.csv passes check.sh format checks;
               frontend calls /run (mock fallback). 
 T2  22:00     REAL v1: extractor + KB + LightGBM labels beat the "all true" baseline on hidden practice
               rows; 1,000 validator-clean claims; frontend shows real results.
 overnight     improvements, calibration, claim diversity, optional neuromorphic experiment (P3).
 T3  08:30 29 Oct  FEATURE FREEZE: fixes, polish, pitch only.
 09:00         `git pull upstream main` → sample/sample.csv appears. Run it in the frontend, record VIDEO.
 10:30         P1 opens the PR (fork → QA-DNA/hype-arena-starter) titled "Tzatziki". 
 11:00         FROZEN. Any push to or closing of the PR after 11:00 disqualifies us.
 17:00         Final pitch (if selected).

TEAMWORK RULES (non-negotiable)
 1. Parallel, never blocked: if a teammate's piece isn't there yet, code against the stub/interface
    and keep going. Never sit waiting for a commit. Never ask a teammate to stop or wait for you.
 2. Stay current, read others' work: at the start of every task run
      git fetch origin && git merge origin/main && git log --oneline -15 origin/main && cat STATUS.md
    and adapt to what teammates merged (new interfaces, new data, requests addressed to you).
 3. ALL WORK IS ON BRANCHES. One branch per task, named <name>/<topic> (X/contracts, ioan/dockerfile,
    arckan/extractor, vald/run-page). Never commit directly to main.
    Commit small and often on your branch (every working step, at least every 30 min), message
    "<P#-role>: <what>" (e.g. "P3-ai: attribute classifier v1, 0.91 acc"), and push the branch after
    every commit (git push -u origin HEAD) so teammates can see progress (git fetch; git log origin/<branch>).
 4. Merge to main often so nobody is held back: as soon as a step works (python -m pytest -q or
    npm run build green, branch up to date with origin/main), open a PR to OUR repo and merge it yourself:
      gh pr create --repo Arkhan12345/HoolyTzatziki-hype-arena-starter --base main --fill
      gh pr merge --merge --delete-branch
    At least every 1-2 h. X reviews merged PRs asynchronously and never blocks a merge. main must
    always run. Need a teammate's unmerged work right now? Ask them to merge it; only as a last resort
    git merge origin/<their-branch> into your branch.
 5. Ownership: edit only your folders. Need something elsewhere? Add a line under "Requests" in
    STATUS.md: "@P2 from P3: need KB.lookup to return history sorted by date". Owners check requests
    every pull and answer by committing.
 6. Living pitch: after each meaningful commit append ONE line to pitch/JOURNAL.md:
      `HH:MM | P# | what was built | why it matters / number`
 7. Update your row in STATUS.md every hour: | time | member | done | doing next | blocked by |
 8. Never commit secrets, .env, large raw files (>20 MB) or anything from outside the repo.
 9. Never use an LLM at runtime; never fetch from the network at runtime.
=== END SHARED BRIEF ===
~~~

---

## PROMPT P4 — Fullstack + Frontend (Vald)

~~~text
ROLE: You are the Claude Code agent of P4 Vald, FULLSTACK + FRONTEND of Team Tzatziki (see Shared Brief
above). In parallel: P1 pushes skeleton + contracts + pitch/PITCH.md, P2 builds the FastAPI
(/run, /architecture, /metrics), P3 builds the model that fills /metrics. Start immediately in
frontend/ with mock data shaped like the Shared Brief's API; switch to the real API when it lands.

I OWN: frontend/   (everything inside it, including its package.json, scripts and mocks)
I DO NOT EDIT: anything outside frontend/. Need an API change? Request to P2 in STATUS.md.
             Need pitch content? It's P1's pitch/PITCH.md — render it, don't rewrite it.

MISSION
The jury sees our system through you. REQUIRED by the rules: a working page that shows the
architecture of our system as a diagram AND runs the system live on rows we give it (the 20-row
sample/sample.csv released at 09:00 on 29 Oct). The 3-minute video is filmed on your UI.

STACK / LOOK
Vite + React + TypeScript + Tailwind + shadcn/ui. Components copied in from Watermelon UI
(https://ui.watermelon.sh) and Cult UI (https://www.cult-ui.com) via their shadcn registry CLI
(`npx shadcn@latest add <registry-url>`) — e.g. animated cards, bento grid, tabs, text animations,
dynamic island / toasts. Diagram: React Flow (@xyflow/react) (or Mermaid as fallback). Charts: Recharts.
Style nod to the challenge brand: sapphire #2859B5, lime #D7F13B, onyx #0A0A0A, paper #FCFCFC;
TRUE/FALSE/UNSURE "stamps" like the brief page. Works at 1920×1080 and on a projector.

TASKS, IN ORDER
1. [T0, first push within ~45 min] Scaffold frontend/ (Vite React TS, Tailwind, shadcn init),
   routes: / (Run), /architecture, /metrics, /pitch. src/api/client.ts with VITE_API_URL
   (default http://localhost:8000) and VITE_MOCK=1 switch; src/mocks/*.json shaped exactly like the
   Shared Brief API ({labels, claims, facts, timings, stages}). Commit.
2. [T0→T1] /architecture: React Flow diagram of the pipeline — Universe CSV → Extractor (gazetteer +
   regex + attribute classifier + coreference) → Temporal KB → Truth classifier (LightGBM +
   calibration) → Labels; KB → Claim generator → Validator → 1,000 claims → output.csv; optional
   CfC/LNN node (dashed). Nodes from GET /architecture when available (mock until then). Clicking a
   node shows what it does + its latest timing. During a run, nodes light up in sequence.
3. [T1] / (Run page): upload CSV or "Use sample.csv" button → POST /run → split view: architecture
   mini-diagram animating + results table (row text, date, TRUE/FALSE/UNSURE stamp, confidence bar,
   extracted facts as chips subject·attribute·value). Tab "Generated claims" (type badge, text,
   form JSON collapsible, confidence). Loading / empty / error states; never a blank screen.
4. [T2] /metrics: label accuracy & macro-F1 vs baseline, confusion matrix, calibration curve,
   claims by type, runtime per stage — from GET /metrics (pitch/metrics.json). Include the
   LEADERBOARD: sortable table from metrics.json "leaderboard" (model, track classic/liquid/spiking,
   features with/without truth discovery, CV scheme, macro-F1, accuracy, ECE, inference time, size,
   ships?) with the shipped model highlighted, filters by track, a "Classic vs Liquid vs Spiking" bar
   chart (best of each track) and a SHAP top-features chart when present.
5. [T2→T3] /pitch: render pitch/PITCH.md (fetched from API or imported at build time via a
   symlink-free copy step in a frontend script) as full-screen slides: each "## Slide N" = one
   slide, arrow keys to navigate, "> notes:" hidden in presenter mode (press N). Pull live numbers
   from /metrics into slides that reference them. This is how the pitch stays current all night.
6. [T3 → 09:00] Demo polish: a "Demo mode" button that runs sample.csv and walks the diagram;
   no console errors; frontend/README.md + scripts start.ps1 / start.sh (install + run API + run
   UI with one command, documented for the PR). Take screenshots for P1 (pitch/), via Request.
7. [09:00 29 Oct] After P1 pulls upstream, run the real sample.csv end-to-end; record the video
   with P1 (diagram + live run + results), ≤ 3 min.

DEFINITION OF DONE: npm run build passes; diagram + live run of sample.csv working against the real
API; metrics + pitch pages show live numbers; one-command start documented.

HOW TO WORK (Claude Code, single agent; sub-agents allowed only inside frontend/)
- Start every task: `git fetch origin && git merge origin/main; git log --oneline -15 origin/main; cat STATUS.md` — check P2's API
  shape changes and P1's PITCH.md updates, adopt them. Never wait for the backend: keep mocks.
- Commit per screen/component ("P4-frontend: …"), `npm run build` before push; branch
  vald/<topic>, push after each commit, merge to main via PR when green. JOURNAL line per visible feature; STATUS hourly.
~~~

---

## Next steps from the research (28 Oct evening)

> From P1's research: the strongest demo moment is the same sentence turning from true to false once the value moves on.

1. **Timeline view per verdict → `frontend/src/` (a new component on `/run`).** Clicking a label opens the fact's timeline: steps over time (date, value, pages, trust), the sentence's date and value marked on it, and the verdict and reason. The data shape is in `eda/out/handoff/timeline_examples.json` (10 real rows); use it as the mock now. The real field comes from P2 as `labels[].timeline`.
2. **Demo buttons.** Next to "Use sample.csv", add "Failure types" (loads `eda/out/handoff/demo_failure_types.csv`, 19 rows, one per failure type, including two same-sentence/different-date pairs) and "Write your own" (`eda/out/handoff/demo_blank.csv`, header only, editable table). Copy the CSVs into `frontend/public/demo/` at build time, like `copy-pitch.mjs` does.
3. **Calibration chart → `/metrics`.** From `eda/out/handoff/calibration_bins.csv` (arrives at integration, `{{RESULT:calibration_bins}}`): bars for share correct per confidence band, next to the diagonal.
4. **Verify.** `npm run build`; click through `/run` with each demo file; the timeline opens for every labelled row (mock now, live once P2 merges the field).
