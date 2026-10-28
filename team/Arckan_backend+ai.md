# Arckan · P3 · Backend + AI (the model) — Team Tzatziki · Hype or Truth

> **How to use:** open Claude Code in your clone of https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter and paste this WHOLE file as the first message (Shared Brief + your role prompt). Read `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/EDA.md`, **`docs/MODELS.md` (your model plan — follow its order)** and `team/README.md` first.

## FILES THIS PROMPT CREATES / MODIFIES
You (Claude) may READ every file in the repo. You may only CREATE or MODIFY these:

```text
eda/eda.py · eda/report.md
app/extract/{gazetteer,values,attributes,coref}.py · app/extract/__init__.py (exports extract)
app/classify/{features,model,calibrate}.py · app/classify/__init__.py (exports classify)
train/train.py · models/*.joblib (< 50 MB total) · requirements-train.txt
tests/model/test_extract.py · tests/model/test_classify.py
app/neuro/{timeline,cfc,bench}.py (OPTIONAL, after T2)
pitch/metrics.json (keys "baseline","extraction","model","calibration","neuro") · pitch/JOURNAL.md (append) · STATUS.md (own row + requests)
```
Anything else → a Request line in STATUS.md to its owner.

## RECOMMENDED CLAUDE CODE SETUP (skills, tools, harness) for this session
Install or enable these before pasting the prompt. Anything marked *(if available)* depends on your account or plugins, so skip it if it's missing; nothing below is required to start.

**Everyone (same baseline on all 4 laptops):**
- **Model:** Opus via `/model`. Effort **high** for design/rules work, **medium** for routine edits.
- **CLIs:** `gh` (logged in with `gh auth login`), `git`, Python 3.12, Docker Desktop. On Windows, run `scripts/check.sh` from Git Bash or WSL.
- **Shared harness:** X commits `.claude/settings.json`, so every session gets the same setup. It holds:
  - a **SessionStart hook** that runs `git pull --rebase --autostash && git log --oneline -10 && cat STATUS.md`, so the agent sees teammates' commits every time it starts;
  - a **permissions allowlist** for `git pull/add/commit/push`, `python -m pytest`, `npm run *`, `docker build/run`, `sh scripts/check.sh` (fewer prompts, faster parallel work). Afterwards, run `/fewer-permission-prompts` to add the rest.
- **Context7 MCP** for up-to-date library docs: `claude mcp add context7 -- npx -y @upstash/context7-mcp`.
- **Built-ins to use:**
  - **Plan mode** (Shift+Tab) before any multi-file change.
  - **Explore subagent** to read contracts/ and teammates' code without flooding context.
  - **`run_in_background`** for long commands (training, docker build, check.sh).
  - **`/code-review`** on your own diff before pushing big changes.
  - **`/loop 60m`** to refresh your STATUS.md row and read new Requests.

**For Arckan (backend + AI):**
- **Built-in notebook editing** (NotebookEdit) if you like notebooks, but keep `eda/eda.py` as the source of truth.
- **`dataviz` skill** *(if available)* for EDA plots and the calibration curve that goes into the pitch.
- **Context7** for scikit-learn, LightGBM, (optional) ncps / snnTorch / torch-CPU.
- **Long jobs:** run training and cross-validation with `run_in_background`, and use `/loop` to poll results and append metrics to `pitch/metrics.json`.
- **Plan subagent** before the feature set and CV design. Grouped CV by page/date is easy to get wrong.
- **Optional neuromorphic arm:** a separate venv (`requirements-train.txt` with torch-CPU + ncps). Have the Explore subagent read `neuromorphic-gestures/learn/_lib/` if that folder exists on your laptop.
- **Hook (your own `.claude/settings.local.json`):** a PostToolUse hook that runs `python -m pytest -q tests/model` after edits in `app/extract/**` or `app/classify/**`.

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
Optional (P3, after T2 only): torch-CPU + ncps (liquid CfC network) behind USE_NEURO=1.

MODEL (agreed): Hybrid temporal Knowledge Base + classic ML
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
 2. Pull often, read others' work: at the start of every task run
      git pull --rebase && git log --oneline -15 && cat STATUS.md
    and adapt to what teammates pushed (new interfaces, new data, requests addressed to you).
 3. Commit small and often: every working step, at least every 30 min. Message: "<P#-role>: <what>"
    (e.g. "P3-ai: attribute classifier v1, 0.91 acc"). Always `git pull --rebase` before `git push`.
 4. Never break main: before pushing run `python -m pytest -q` (backend) or `npm run build`
    (frontend). Big/risky changes → branch feat/<role>-<topic>, merge to main once green
    (rebase, then fast-forward). Delete the branch after.
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

## PROMPT P3 — Backend + AI: the model (Arckan)

~~~text
ROLE: You are the Claude Code agent of P3 Arckan, BACKEND + AI (THE MODEL) of Team Tzatziki
(see Shared Brief above). In parallel: P1 pushes the skeleton/contracts.py, P2 builds the Docker
runtime, the temporal KB (app/kb) and the claim generator, P4 builds the frontend showing your
results. Start now on EDA + extractor in your own folders; adopt contracts.py when it lands.

I OWN: app/extract/ app/classify/ app/neuro/ models/ eda/ train/ tests/model/ requirements-train.txt
I DO NOT EDIT: app/run.py, app/kb, app/generate, app/validate, app/api, Dockerfile (P2 — send
       Requests, e.g. extra pip packages), frontend/ (P4), docs/ pitch/PITCH.md CLAUDE.md
       app/contracts.py (P1), contracts/ data/ sample/.
RULE: the shipped system is NOT an LLM and uses no network. You MAY use an LLM (yourself) during
development to write code, rules, patterns, or synthetic training text — saved as files in the repo.

MODEL PLAN: docs/MODELS.md is the agreed series of models (NLP extraction → truth discovery →
LightGBM + LogReg + isotonic; then CfC/LTC/GRU feature; SNN + LSM benchmark). Follow its milestone order
and fill its comparison table into pitch/metrics.json.

KEY EDA FACTS (docs/EDA.md)
1,897 practice rows: true 83%, false 15%, unsure 2%. 62 subjects, 321 attributes, 13 events.
Value kinds: quantity 54%, money, duration, name, percent, boolean, date, version.
329/514 (subject,attribute) pairs change value over time → truth is date-dependent.
196/360 false facts contradict the latest prior true value → a temporal KB explains most falses.
754 facts don't name their subject in the text (He / Its / The platform) → page coreference.
Unsure is rare and mostly about known pairs → prefer true/false with honest confidence.
Universe input has NO form column: we must extract facts from text ourselves.

TASKS, IN ORDER
1. [T0→T1, first commit within ~45 min] eda/eda.py (script, not only notebook) reproducing the
   numbers above + label rate by kind/attribute/page, top lying pages (cue words e.g. "Gavin:"),
   value timelines per pair. Write eda/report.md. Baselines into pitch/metrics.json: "all true"
   accuracy, TF-IDF+LogReg text-only accuracy (grouped CV by date).
2. [T0→T1] app/extract v0 — returns contract Facts:
   a. subject gazetteer from lexicon names+aliases (whole words, case-insensitive, longest match
      wins: "Hooly AI Pro" beats "Hooly").
   b. value regexes per kind following contracts/value-forms.md (money w/ currency+period, quantity
      with unit noun, percent, duration units, dates with year, versions, boolean via negation words);
      skip approximations/ranges/comparisons as value-forms.md says ("over 100" = op gt).
   c. attribute classifier: TF-IDF (word+char n-grams of the sentence with the value masked) +
      value kind + subject's allowed attributes from the lexicon → LogisticRegression trained on
      practice `form`s. Restrict prediction to attributes the subject has.
   d. coreference: if no subject named, take the most recent subject on the same page (same date,
      nearby ids) whose attribute set fits the value kind.
   Measure extraction precision/recall vs gold forms (grouped CV) → metrics.json "extraction".
   Push v0 early even if weak — P2's KB and P4's UI need real facts flowing.
3. [T1→T2] app/classify — features per row (via P2's KB helpers; until they land, compute your own
   in-module from gold forms): matches value in force at row date, contradicts it, value newer than
   anything known, n agreeing / disagreeing sources across universe, page trust, page false-rate cue
   words, value kind, n facts, extraction confidence, has comparison op, speaker-quote markers.
   Model: LightGBM (fallback LogisticRegression), grouped CV by page/date on practice.
   IMPORTANT for honest eval: when evaluating, build the KB WITHOUT the held-out fold's labels
   (check.sh hides every 5th row and our image contains practice.csv, so its score is optimistic).
4. [T2] Calibration: isotonic / Platt on out-of-fold predictions → confidence = P(predicted label) in
   [0.5, 1]. "unsure" only when max prob < threshold tuned on OOF (rare). Calibration curve + ECE
   into metrics.json "calibration" (P4 plots it, P1 pitches it).
5. [T2] Train script train/train.py → models/*.joblib (small, < 50 MB total); inference on 20k rows
   < 5 min on 2 CPUs (time it, report in metrics.json). Tests in tests/model/ (shapes, determinism,
   confidence range). Tell P2 via Requests exactly what pip packages/versions inference needs.
6. [T2 → T3] Error analysis loop: top confusions → new regexes / features → retrain. Help P2's
   generator by exposing `confidence_for_fact(subject, attr, value, date)` so only high-confidence
   facts become claims.
7. [OPTIONAL, only after T2 is green, timebox 2 h] NEUROMORPHIC EXPERIMENT in app/neuro/:
   Reference code (read-only, never import from it — copy minimal pieces into app/neuro/):
   C:\Users\alexandru.mirea\PycharmProjects\neuromorphic-gestures
     learn/_lib/models.py  build("cfc"|"ltc"|"gru"|"snn_mlp", in_shape, n_classes), SeqNet
     learn/_lib/trainer.py fit/predict ; learn/_lib/common.py rate_encode/latency_encode/snn_classifier
     learn/_lib/compare.py benchmark tables
   (Not on that laptop? It's a liquid-network/SNN lesson repo using ncps CfC + snnTorch; write the
   ~60 lines yourself with `pip install ncps` + torch CPU.)
   Idea: for each (subject, attribute) build an irregular-time event sequence (Δdays, agrees-with-KB,
   page trust, kind) → CfC predicts "value still current at date t" → one extra LightGBM feature
   `cfc_current_prob`. Spiking MLP (snnTorch) over the tabular features = benchmark row only.
   Ship behind USE_NEURO=1 ONLY if grouped CV improves AND inference < 2 min AND torch-CPU fits the
   image (ask P2 to add `torch --index-url https://download.pytorch.org/whl/cpu` + ncps). Either way
   write the comparison table (Classic vs Liquid vs Spiking: accuracy, F1, ECE, latency, params) to
   metrics.json "neuro" — P4 shows it, P1 pitches it honestly.

DEFINITION OF DONE: extract + classify plugged into app/run.py via contracts; beats baseline by a
clear margin in grouped CV; calibrated confidence; models/ committed; metrics.json filled.

HOW TO WORK (Claude Code, single agent; sub-agents allowed only inside my folders)
- Start every task: `git pull --rebase; git log --oneline -15; cat STATUS.md` — read P2's KB API and
  P1's contracts changes; answer Requests to me. Never wait: stub what you need locally.
- Commit per step ("P3-ai: …") with the metric in the message; pytest before push; pull --rebase
  before push; experiments on feat/ai-<topic>. JOURNAL line per meaningful result; STATUS hourly.
~~~
