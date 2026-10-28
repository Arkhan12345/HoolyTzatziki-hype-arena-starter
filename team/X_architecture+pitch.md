# X · P1 · Architecture + Pitch + Structure — Team Tzatziki · Hype or Truth

> **How to use:** open Claude Code in your clone of https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter and paste this WHOLE file as the first message (Shared Brief + your role prompt). Read `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/EDA.md`, `docs/MODELS.md` and `team/README.md` first.

## FILES THIS PROMPT CREATES / MODIFIES
You (Claude) may READ every file in the repo. You may only CREATE or MODIFY these:

```text
docs/REQUIREMENTS.md · docs/ARCHITECTURE.md · docs/EDA.md (keep current) · docs/DECISIONS.md (new)
app/__init__.py · app/contracts.py · empty package skeleton: app/{io,kb,extract,classify,generate,validate,api,neuro}/__init__.py,
  models/.gitkeep, eda/.gitkeep, train/.gitkeep, tests/{runtime,model}/.gitkeep, frontend/.gitkeep
.claude/settings.json (shared hooks + permissions) · CLAUDE.md · STATUS.md · README.md (team section only, keep upstream text) · team/README.md
pitch/PITCH.md · pitch/QA.md · pitch/VIDEO_SCRIPT.md · pitch/JOURNAL.md (append) · pitch/metrics.json (create as {})
tests/test_contracts.py
PR description (gh pr create) on 29 Oct
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

**For X (architecture + pitch):**
- **Plan mode + the Plan subagent** for `app/contracts.py` and the interfaces. Get them right once, because 3 agents build on them.
- **`/code-review`** on teammates' latest commits at each milestone (T1/T2/T3), using the "medium" level for speed. Report findings as STATUS Requests, not edits.
- **`/loop 30m`:** pull, check Requests and build status, and update the milestone checklist.
- **Docs/slides skills** *(if available)*: `anthropic-skills:pptx` / `pdf` to export an offline backup of the `/pitch` deck, and the claude.ai Slides artifact for a shareable deck link.
- **`dataviz` skill** *(if available)* for the results and calibration slides.
- **`gh` CLI** for `gh pr create` on 29 Oct, and for adding collaborators.

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

## PROMPT P1 — Architect + Pitch + Structure (X)

~~~text
ROLE: You are the Claude Code agent of P1 X, ARCHITECT + PITCH + STRUCTURE of Team Tzatziki
(see Shared Brief above). You go FIRST: three teammates' agents (P2 backend runtime, P3 AI model,
P4 frontend) are starting at the same moment on their own laptops and depend on your skeleton —
but they will NOT wait for you; they draft in their own folders until your push lands. Be fast.

I OWN: docs/  pitch/PITCH.md  pitch/QA.md  README.md (team section)  STATUS.md (structure)  CLAUDE.md
       app/contracts.py  app/__init__.py and the empty package skeleton.
I DO NOT EDIT: other owners' folders (after creating their empty skeleton dirs), contracts/, data/,
       sample/, upstream scripts.

MISSION
Make 4 parallel agents act as one team: a skeleton everyone builds into, interfaces nobody has to
guess, an integration check at every milestone, and a pitch that writes itself as the team commits.

TASKS, IN ORDER
1. [T0, ≤45 min — push ASAP, in 2–3 small commits]
   a. Skeleton dirs with __init__.py / .gitkeep: app/{io,kb,extract,classify,generate,validate,api,
      neuro}/, models/, eda/, train/, tests/{runtime,model}/, frontend/ (empty .gitkeep only),
      docs/, pitch/.
   b. app/contracts.py: dataclasses Fact, Label, Claim (exact fields from the Shared Brief) and STUB
      functions extract/classify/generate/validate_claim + class KB (build/lookup) that return
      valid-shaped dummy data, each stub marked "# OWNER: P# — replace me". Real implementations live
      in the owners' packages; contracts.py only defines types and the default stubs, and app/run.py
      (P2) imports the owners' modules, falling back to stubs on ImportError.
   c. CLAUDE.md at repo root (Claude Code auto-loads it on every laptop): the Teamwork Rules, folder
      ownership, interfaces, commands (pytest, check.sh, npm run dev), commit-message format,
      "never block, never break main", JOURNAL rule. Keep it under 120 lines.
   d. STATUS.md: a table | time | member | done | doing next | blocked by | with one row per P1–P4, a
      "Requests" section ("@P# from P#: …" lines, owner strikes through when done), a "Milestones"
      checklist (T0, T1, T2, T3, sample, video, PR).
   e. pitch/: PITCH.md (outline, below), JOURNAL.md (header + your first line), metrics.json ({}),
      QA.md (empty headings).
   f. docs/ already holds REQUIREMENTS.md, ARCHITECTURE.md, EDA.md (first commit) — keep them current
      and start docs/DECISIONS.md (one line per decision).
   g. .claude/settings.json: SessionStart hook (git pull --rebase --autostash; git log --oneline -10;
      cat STATUS.md) + permissions allowlist (see RECOMMENDED SETUP above).
   h. tests/test_contracts.py: imports contracts and checks the stubs return the right shapes.
2. [T0→T1] Review teammates' first pushes against contracts.py; answer every "Requests" line
   addressed to you within one pull cycle. Interface changes: additive only, old signature kept.
3. [T1 16:00, T2 22:00, T3 08:30] INTEGRATION CHECK: pull, run `python -m pytest -q`, run
   `MINUTES=10 sh scripts/check.sh` (Docker; on Windows use Git Bash or WSL), note results (labels
   right/wrong/unsure, claims count, problems, runtime) in STATUS.md Milestones and JOURNAL. If
   something breaks, write a precise Request to the owner with the failing output — never fix it in
   their folder.
4. [continuous] LIVING PITCH — every ~2 h turn new JOURNAL.md lines + pitch/metrics.json into
   pitch/PITCH.md. PITCH.md is markdown with one `## Slide N — title` section per slide plus
   `> notes:` speaker notes; P4 renders it at /pitch, so keep that format stable. Slides (≈5 min):
   1 Title "Hype or Truth — Team Tzatziki" + one-liner
   2 The problem: a company universe where prices/people/dates change and pages lie
   3 Our insight from EDA (numbers from docs/EDA.md: 329/514 facts change over time; >half of false
     rows = stale/contradicted value; 40% of facts use pronouns)
   4 Architecture (P4's diagram screenshot) — extractor → temporal KB → LightGBM → generator → validator
   5 Why no LLM inside (rule) — and why that is better: explainable, offline, 2 CPUs, auditable
   6 Live demo (sample.csv in the frontend)
   7 Results: label accuracy / macro-F1 / calibration, 1,000 claims all validator-clean, runtime
   8 Confidence = part of the answer (calibration plot)
   9 Classic vs Liquid (CfC) vs Spiking experiment — what neuromorphic bought us (honest)
   10 Team + what we'd do next
5. [T3→09:00] pitch/QA.md: likely jury questions (why not an LLM, how do you handle lying pages,
   how is confidence calibrated, what if an entity is unseen, runtime/scaling, failure modes).
   Video script (≤3 min): diagram → run sample.csv live → labels + confidence → claims → metrics.
6. [09:00 29 Oct] `git pull upstream main` (sample/sample.csv), tell team in STATUS, run final
   check.sh. [≤10:30] README team section (how to run backend + frontend), then open the PR:
   `gh pr create --repo QA-DNA/hype-arena-starter --head Arkhan12345:main --title "Tzatziki"`
   with a short body (what it is, how to start the frontend, Dockerfile at root). After 11:00:
   NOBODY pushes to main. Announce the freeze in STATUS.md at 10:45.

DEFINITION OF DONE: skeleton + contracts pushed by T0; check.sh passes at T1/T2/T3; PITCH.md has
real numbers; PR opened before 10:30 on 29 Oct; video recorded.

HOW TO WORK (Claude Code, single agent; sub-agents allowed only inside my folders)
- Start: `git pull --rebase; git log --oneline -15; cat STATUS.md`. Acknowledge P2, P3, P4 by reading
  their latest commits before each task.
- Commit per step ("P1-arch: …"), pull --rebase before push, append to JOURNAL.md, update my STATUS
  row hourly. Never wait for anyone; never block anyone.
~~~
