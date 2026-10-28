# Ioan · P2 · Backend + DevOps (runtime, Docker, KB, claim generator) — Team Tzatziki · Hype or Truth

> **How to use:** open Claude Code in your clone of https://github.com/Arkhan12345/HoolyTzatziki-hype-arena-starter and paste this WHOLE file as the first message (Shared Brief + your role prompt). Read `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/EDA.md` and `team/README.md` first.

## FILES THIS PROMPT CREATES / MODIFIES
You (Claude) may READ every file in the repo. You may only CREATE or MODIFY these:

```text
Dockerfile · .dockerignore · .gitignore (additions) · requirements.txt
app/run.py · app/io/{reader,writer}.py · app/kb/{store,normalize,features}.py
app/generate/{templates,render,select}.py · app/validate/rules.py · app/api/main.py
scripts/dev_api.sh · scripts/dev_api.ps1 · scripts/synth_universe.py (new files only — never edit upstream check.sh / input.py / check_output.py)
tests/runtime/test_run.py · tests/runtime/test_generate.py · tests/runtime/test_validate.py
pitch/metrics.json (key "runtime" only) · pitch/JOURNAL.md (append) · STATUS.md (own row + requests)
```
Anything else → a Request line in STATUS.md to its owner.

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

## PROMPT P2 — Backend + DevOps (Ioan)

~~~text
ROLE: You are the Claude Code agent of P2 Ioan, BACKEND + DEVOPS (runtime, Docker, KB, generator) of Team Tzatziki
(see Shared Brief above). P1 (architect) is pushing the skeleton + app/contracts.py in the first
45 min, P3 builds the extractor + classifier, P4 builds the frontend that calls your API — all in
parallel right now. Do not wait for P1's skeleton: start on the Dockerfile and run.py in your own
paths immediately; adopt contracts.py the moment it lands (git pull --rebase).

I OWN: Dockerfile .dockerignore requirements.txt app/run.py app/io/ app/kb/ app/generate/
       app/validate/ app/api/ scripts/ (new files only) tests/runtime/
I DO NOT EDIT: app/extract, app/classify, app/neuro, models/ (P3), frontend/ (P4), docs/, pitch/PITCH.md,
       CLAUDE.md, app/contracts.py (P1), contracts/ data/ sample/ and upstream scripts.

MISSION
The thing QA DNA actually runs. It must ALWAYS produce a valid /output/output.csv within 30 min,
offline, as user 10001 — even if the model crashes. And it must write 1,000 claims that are true,
new and phrased exactly as contracts/claim-forms.md + value-forms.md read them.

TASKS, IN ORDER
1. [T0, first push within ~45 min] Dockerfile: python:3.12-slim, linux/amd64, pip install --no-cache
   from requirements.txt (pandas, numpy, scikit-learn, lightgbm, jsonschema, joblib, fastapi, uvicorn
   — pin versions), COPY app/ models/ data/ contracts/ ; ENV HOME=/tmp XDG_CACHE_HOME=/tmp
   HF_HOME=/tmp MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 ; CMD python -m app.run ;
   last line USER 10001. .dockerignore (frontend/, eda/, .git, node_modules, *.ipynb).
   Dummy app/run.py: read universe.csv, label every blank row "true",0.6, write 1,000 placeholder
   claims → passes `sh scripts/check.sh` format checks. Commit + JOURNAL line.
2. [T0→T1] app/io: robust CSV read (csv module, utf-8-sig, any size), writer (csv.QUOTE_MINIMAL,
   json.dumps(form, ensure_ascii=False)), write to /output/.output.tmp then os.replace.
   app/run.py pipeline: load → extract (P3) → KB.build → classify (P3) → generate → validate → write.
   Safety: (a) write a FALLBACK table within 60 s (majority label + KB-free claims);
   (b) SIGTERM handler + watchdog thread reading ARENA_DEADLINE: at deadline-90 s write best-so-far;
   (c) every stage in try/except → fall back to the stub for that stage, log to stderr;
   (d) import P3 modules with fallback to contracts.py stubs; (e) log stage timings + RSS memory.
   tests/runtime/: run main() on a tmp copy of practice data (via scripts/input.py) and assert the
   output passes scripts/check_output.py.
3. [T1→T2] app/kb — temporal knowledge base:
   - gold facts from data/practice.csv `form` (label true → trusted; label false → known-wrong values)
   - extracted facts from P3's extract() over the whole universe (with confidence)
   - per (subject, attribute): timeline of (date, value, support, sources, trust); value equality via
     normalized values (unit conversion for duration/quantity like value-forms.md "Units convert")
   - per-page trust (page = same date + adjacent ids): share of its facts agreeing with the KB
   - lookup(subject, attr, date) → value in force on that date (latest prior well-supported value),
     support, history. Also current_value(subject, attr, as_of="2026-10-29").
   Expose features helpers P3 can call (agrees_with_latest, n_supporting_sources, value_changed_after).
4. [T1→T2] app/generate — 1,000 claims, ALL TRUE, diversity across types/subjects:
   - Only use facts with high support (gold true or ≥2 agreeing sources and no newer contradiction).
   - fact: current value on 29 Oct 2026 — "<Subject> <attribute phrase> <value>." no other value,
     no date/time word, no negation (except boolean, where the attribute words must appear).
   - change: consecutive distinct values: "<Subject> went from <v1> to <v2>." (+ "change" may say when)
   - at_event: value on an event's date (event dates from KB 'date' attribute of event subjects).
   - compare: two subjects, same kind and comparable unit/currency/period, current values, phrases
     from the claim-forms table ("more … than", "before", "the same … as"), NO numbers/dates in text.
   - relation: link attributes whose value is a name of another subject (e.g. Hooly.cto → Bertram
     Gilfoil) → "The <link words> of <Subject> <attribute phrase> <value>."
   - Value rendering exactly per value-forms.md (money "€59 per month", dates "1 July 2026",
     quantity with unit noun, percent "99.9%", version, duration).
   - Confidence = support-based (0.95 gold-backed, 0.8 multi-source, never below 0.6 → else skip).
   - Dedupe against all input texts (normalized like check_output.py) and against each other; ids
     "tz-0001"…; if short of 1,000, fill with more fact/change claims of next-best support.
5. [T2] app/validate: jsonschema against contracts/entry.schema.json (+claimed_value schema), subject
   names/aliases + attribute exist in lexicon, text names the subject, no negation words (unless
   boolean), no refused date/time words, no second value of same kind, ≤300 chars, not a copy.
   Drop invalid ones and regenerate. Unit tests from contracts/examples/entry/{good,bad}.
6. [T1 stub → T2 real] app/api (FastAPI): POST /run (CSV upload or {"use":"sample"} reading
   sample/sample.csv, else 20 random practice rows) → {labels, claims, facts, timings, stages};
   GET /architecture (stage graph JSON: nodes, edges, descriptions); GET /metrics (pitch/metrics.json
   + runtime metrics); CORS for http://localhost:5173. scripts/dev_api.sh / .ps1 to start it.
   Push the endpoint shapes to STATUS Requests for P4 ASAP (even before they're real).
7. [overnight → T3] Full-size test: synthesize a 20k-row universe (practice ×10 with new ids) and
   confirm run < 10 min, RSS < 3 GB, image < 1.5 GB (`docker image inspect`). Write runtime numbers
   to pitch/metrics.json under key "runtime".

DEFINITION OF DONE: check.sh passes with real pipeline; 1,000 validator-clean claims; run survives a
crashing model and a SIGTERM; API serves the frontend; numbers in metrics.json.

HOW TO WORK (Claude Code, single agent; sub-agents allowed only inside my folders)
- Start every task: `git pull --rebase; git log --oneline -15; cat STATUS.md` — read P1's contracts,
  P3's extractor progress, P4's requests to me. Build on them; never wait for them.
- Commit per working step ("P2-backend: …"), pytest before push, pull --rebase before push.
  Big refactors on feat/backend-<topic>. JOURNAL line per meaningful commit; STATUS row hourly.
~~~
