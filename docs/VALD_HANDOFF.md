# Handoff to Vald (P4): connect the frontend to the real backend

From Ioan (P2). Everything below is implemented on branch `john` and covered by `tests/runtime/test_api.py`.
Exact types: [`docs/API_CONTRACT.md`](API_CONTRACT.md). Schemas in code: `app/metrics/schemas.py`.

## 1. The pipeline in one picture

```
universe rows ─► Extractor (Arckan, app/extract) ─► Temporal KB (Ioan, app/kb) ─► Truth classifier (Arckan, app/classify) ─► labels
                                                     └─► Claim generator + validator (Ioan, app/generate, app/validate) ─► 1,000 claims
labels + claims ─► Writer (Ioan, app/run.py) ─► output.csv          Demo API (Ioan, app/api) ─► your React app
```
Stage ids, in order, used everywhere (`stages`, `timings`, node `stage`): `load, extract, kb, classify, generate, validate, write`.

## 2. Start the backend

Once (from the repo root, Python 3.12):
```powershell
py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```
Every time:
```powershell
scripts\dev_api.ps1                      # Windows: http://127.0.0.1:8000 (uses .venv if present)
```
```sh
sh scripts/dev_api.sh                    # macOS / Linux / Git Bash (python on PATH must have requirements.txt)
```
Your existing `frontend/scripts/start.ps1` / `start.sh` also starts it (`python -m uvicorn app.api.main:app --port 8000`);
that works if the `python` on PATH has `requirements.txt` installed. Check: `curl http://localhost:8000/health`.

| Env var | Side | Default | Meaning |
| --- | --- | --- | --- |
| `VITE_API_URL` | frontend | `http://localhost:8000` | API base URL (already in `src/api/client.ts`) |
| `PORT` | backend (`dev_api.*`) | `8000` | API port |
| `TZ_CORS_ORIGINS` | backend | `http://localhost:5173,http://127.0.0.1:5173` | allowed origins, comma separated |

## 3. Endpoints (all real; nothing below is a mock)

| Use it for | Call | You get |
| --- | --- | --- |
| Is the API up, is the model in? | `GET /health` | `{ok, version, extractor_available, classifier_available, sample_available, runs_in_memory}` |
| Live run on the sample | `POST /run` body `{"use":"sample"}` | `RunResult` |
| Live run on an uploaded CSV | `POST /run` multipart field `file` | `RunResult` |
| Live run on rows you hold | `POST /run` body `{"rows":[{"id","text","date"}]}` | `RunResult` |
| Same, without blocking | `POST /run?mode=async` → 202 `{run_id, poll}`; then `GET /runs/{run_id}` until 200 | `RunResult` |
| All 1,000 claims of a run | `GET /runs/{run_id}/claims?offset=0&limit=1000` (`&type=compare` to filter) | `{run_id, total, offset, limit, claims}` |
| Metrics page | `GET /metrics` (latest run) or `GET /runs/{run_id}/metrics` | `Metrics` |
| Diagram | `GET /architecture` | `{nodes, edges, stage_status, timings, run_id}` (React Flow ready) |
| Download the run's table | `GET /runs/{run_id}/output.csv` | official `output.csv` format |
| Claim validator report | `GET /runs/{run_id}/validation?verdict=PASS\|FAIL\|UNVERIFIED&offset=0&limit=100` | `{selected: {n, verdicts, by_type, checks_failed}, candidates: {generated, verdicts_by_type, rejected, distinct_semantic, semantic_duplicates: {inverse_compare, paraphrase_or_alias}}, total, claims: [{id, type, text, confidence, verdict, checks, reasons, evidence, semantic_key}]}` |
| History | `GET /runs` | last 20 runs |

## 4. Real examples (captured from the running API)

`POST /run?claims_limit=2` with `{"rows":[{"id":"p-14","text":"The programme packed 40 sessions into the schedule.","date":"2026-06-12"},{"id":"x-1","text":"Hooly AI Pro costs €59 per month.","date":"2026-10-01"}]}`:
```json
{
  "run_id": "20261028-123025-fe8573", "status": "ok", "source": "json rows", "created_at": "2026-10-28T12:30:33+0200",
  "rows": [{"id": "p-14", "text": "The programme packed 40 sessions into the schedule.", "date": "2026-06-12"},
           {"id": "x-1", "text": "Hooly AI Pro costs €59 per month.", "date": "2026-10-01"}],
  "labels": [{"id": "p-14", "label": "true", "confidence": 0.83, "gold": "false"},
             {"id": "x-1", "label": "true", "confidence": 0.83, "gold": null}],
  "predictions": [{"id": "p-14", "text": "The programme packed 40 sessions into the schedule.", "date": "2026-06-12",
                   "label": "true", "confidence": 0.83, "gold": "false", "facts": [], "evidence": null,
                   "explanation": "No classifier yet (P3): the label is the all-true baseline (83% of practice rows are true)."}],
  "facts": [],
  "claims": [{"id": "tz-0001", "type": "compare", "text": "The joined of Bertram Gilfoil is after the joined of Holdan.",
              "form": {"left": {"subject": "Bertram Gilfoil", "attribute": "joined"},
                       "right": {"subject": "Holdan", "attribute": "joined"}, "op": "gt"}, "confidence": 0.97}],
  "claims_total": 1000, "claims_by_type": {"compare": 510, "fact": 181, "at_event": 28, "change": 281},
  "timings": {"load": 0.0, "extract": null, "kb": 0.0113, "classify": null, "generate": 0.3624, "validate": 7.796, "write": null},
  "stages": ["load", "extract", "kb", "classify", "generate", "validate", "write"],
  "stage_details": [{"id": "extract", "seconds": null, "status": "not_available"},
                    {"id": "generate", "seconds": 0.3624, "status": "ok", "note": "generated"}],
  "metrics": {"…": "see below"}, "errors": []
}
```
(`predictions` and `stage_details` shortened here; the API returns one entry per row and per stage.)

`GET /metrics` after that run (shortened):
```json
{
  "schema_version": "1.0", "run_id": "20261028-123025-fe8573", "kind": "demo", "status": "ok", "errors": [],
  "classification": {"rows_total": 2, "rows_labelled_in_input": 0, "rows_predicted": 2,
    "predicted": {"true": 2, "false": 0, "unsure": 0},
    "confidence": {"n": 2, "min": 0.83, "max": 0.83, "mean": 0.83, "median": 0.83, "histogram": [{"lo": 0.8, "hi": 0.85, "count": 2}]},
    "classifier": "baseline",
    "evaluation": {"label_source": "data/practice.csv", "n": 1, "accuracy": 0.0, "macro_f1": 0.0,
      "baseline": {"name": "all true", "accuracy": 0.0, "macro_f1": 0.0},
      "confusion": {"labels": ["true", "false", "unsure"], "matrix": [[0, 0, 0], [1, 0, 0], [0, 0, 0]]},
      "calibration": [{"predicted": 0.83, "observed": 0.0, "count": 1}], "ece": 0.83,
      "note": "demo rows that are practice rows; their gold was held out of the KB for this run"}},
  "claims": {"candidates_total": 10711, "candidates_by_type": {"compare": 8442, "change": 414, "fact": 848, "at_event": 982, "relation": 25},
    "valid": 10017, "unique_text": 10017, "unique_semantic": 5652, "semantic_duplicates": 4365,
    "selected": 1000, "selected_by_type": {"compare": 510, "fact": 181, "at_event": 28, "change": 281},
    "selected_semantic_duplicates": 0, "gold_only_supported": 1000, "kb_supported": 0,
    "confidence": {"n": 1000, "min": 0.93, "max": 0.97, "mean": 0.949, "median": 0.95, "histogram": ["…"]},
    "rejected_by_reason": {"compare": {"a time word in a compare": 274}}, "inventory": {"…": "…"}},
  "runtime": {"total_s": 8.171, "stages": {"load": 0.0, "extract": null, "kb": 0.0113, "classify": null,
    "generate": 0.3624, "validate": 7.796, "write": null},
    "stage_status": {"extract": "not_available", "classify": "not_available", "write": "skipped", "…": "ok"}, "rss_mb": null},
  "model": {"accuracy": 0.0, "macro_f1": 0.0, "baseline": {"…": "…"}, "confusion": {"…": "…"}, "calibration": [],
            "n": 1, "label_source": "data/practice.csv", "note": "…"},
  "claims_by_type": {"compare": 510, "fact": 181, "at_event": 28, "change": 281},
  "offline": {"source": "pitch/metrics.json (…)", "data": {"runtime": {"…": "…"}}}
}
```

`GET /architecture` (two nodes shown): the nodes of `app/api/architecture.json` (your mock's layout) plus `status`
and `seconds` from the latest run:
```json
{"nodes": [{"id": "extractor", "label": "Extractor", "kind": "stage", "stage": "extract", "owner": "P3", "description": "…",
            "details": ["Gazetteer", "Value regexes", "Attribute classifier", "Coreference"], "position": {"x": 260, "y": 160},
            "status": "not_available", "seconds": null},
           {"id": "kb", "label": "Temporal KB", "kind": "stage", "stage": "kb", "owner": "P2", "description": "…",
            "position": {"x": 520, "y": 160}, "status": "ok", "seconds": 0.0113}],
 "edges": [{"source": "universe", "target": "extractor"}, {"source": "extractor", "target": "kb"}],
 "stage_status": {"load": "ok", "extract": "not_available", "kb": "ok", "classify": "not_available", "generate": "ok", "validate": "ok", "write": "skipped"},
 "timings": {"load": 0.0, "extract": null, "kb": 0.0113, "classify": null, "generate": 0.3624, "validate": 7.796, "write": null},
 "run_id": "20261028-123025-fe8573"}
```

## 5. The 20-row sample (29 Oct, 09:00)
1. `git pull upstream main` on the branch the demo runs from: `sample/sample.csv` appears.
2. Restart the API (or not: it reads the file on every call). `GET /health` → `"sample_available": true`.
3. Your "Use sample.csv" button already sends `POST /run {"use":"sample"}`: the response's `source` is `"sample/sample.csv"`.
   Before 09:00 the same call runs 20 random practice rows and says so in `source` (`"practice (random 20)"`).
4. The sample's labels are unknown: `gold` is `null` on every row and `metrics.classification.evaluation` is `null`. Show
   the predictions, not an accuracy.

## 6. Instructions for Vald's Claude Code Agent

You build the UI only. The pipeline, labels, claims and every number come from this API. Do not compute labels,
claims or metrics in the browser, and do not add a pipeline under `frontend/`.

1. **Keep `src/api/client.ts`'s calls**: `POST /run` (`{use:"sample"}` or multipart `file`), `GET /architecture`,
   `GET /metrics`. They match the backend. Mocks stay as the offline fallback only, with the "API offline" banner.
2. **Extend `src/api/types.ts`** with what the backend adds (all already returned):
   - `RunResponse`: `run_id: string`, `status`, `source: string`, `predictions: Prediction[]`, `claims_total: number`,
     `claims_by_type: Record<string, number>`, `stage_details`, `metrics: Metrics`, `errors`; `timings` values are
     `number | null`; `Label.gold?: Verdict | null`.
   - `ArchNode`: `status?: StageStatus`, `seconds?: number | null`; `Architecture`: `stage_status`, `timings`, `run_id`.
   - `Metrics`: `classification`, `claims`, `runtime.stage_status`, `offline`. Copy the shapes from `docs/API_CONTRACT.md`.
3. **Live run page** (`/`): render `predictions` (text, date, label stamp, confidence bar, `explanation`, `facts` as chips,
   `evidence` when not null). Show `gold` next to the stamp only when it is not null (practice rows). Show
   `claims_by_type` and the first claims from `claims`; for the full list call
   `GET /runs/{run_id}/claims?offset={o}&limit=100` (page) or `&type=compare` (filter tabs). `claims_total` is the count.
4. **Diagram**: light nodes by `node.status` (`ok` green, `fallback` amber, `not_available` grey, `skipped` muted,
   `idle` default) and label them with `node.seconds` (`null` → no number). After a run, refetch `GET /architecture`
   or use the run's `stage_details`.
5. **Metrics page**: `model` may be `null` (no labelled rows). Then show "not available" instead of accuracy,
   macro-F1, confusion and calibration; never fall back to the mock numbers when the API answered. When
   `model.label_source` is `"offline evaluation (pitch/metrics.json)"`, label the card "offline evaluation".
   Claims: `metrics.claims` (candidates, rejected per reason, distinct, selected by type, confidence). Runtime:
   `metrics.runtime.stages` (seconds, `null` = did not run) and `stage_status`.
6. **Unavailable values**: any `null` → "—" with a tooltip "not computed in this run"; `stage_status: "not_available"`
   → "waiting for Arckan's model". Never replace `null` with 0.
7. **Loading, errors, timeouts**:
   - A sync `POST /run` takes about 8 s the first time (it generates and validates ~10,000 candidate claims) and
     well under 1 s when the claims are cached (same held-out rows again). Set the fetch timeout to 60 s and keep the
     stage animation going meanwhile.
   - For a non-blocking UI use `POST /run?mode=async` (202 `{run_id, poll}`), then poll `GET /runs/{run_id}` every
     500 ms: 202 = still running, 200 = done, 500 = `{status:"failed", error}`.
   - 400 = no rows / bad CSV (show the `detail`), 413 = more than 2,000 rows, 404 on `/runs/...` = the API restarted
     (it keeps the last 20 runs in memory): start a new run.
   - `status: "partial"` + `errors[]` = a stage fell back (e.g. the classifier crashed and labels are the baseline):
     show the result with a warning listing `errors[].stage`.
8. **Do not modify**: `app/**`, `contracts/**`, `data/**`, `sample/**`, `scripts/check.sh`, `scripts/input.py`,
   `scripts/check_output.py`, `Dockerfile`, `requirements*.txt`, `tests/runtime/**`. `app/api/architecture.json`
   started as your mock and drives `/architecture`: to change labels/positions, put the new JSON in a STATUS.md Request
   to P2 (or a commit to `frontend/src/mocks/architecture.json` that P2 copies over).
9. **Dependencies**: read `DEPENDENCIES.md` before adding any npm package; frontend versions are pinned by
   `frontend/package-lock.json`.

Validator numbers are also in `metrics.claims.validation` (`candidates_verdicts_by_type`, `inverse_compares_detected`,
`paraphrases_detected`, `selected`). The arena run writes the same report to `validation_report.json` next to
`metrics.json` (`$TZ_AUX_DIR`, default `/tmp/tzatziki` in the arena, next to `output.csv` locally).

## 7. What works now, what is still stubbed

| Piece | State |
| --- | --- |
| API, runs store, claims paging, async runs, output.csv download | works |
| Claim generation (1,000 true, validated claims from the practice gold) | works |
| Metrics (classification counts, claims, runtime; evaluation on practice rows) | works |
| Architecture with live status/seconds | works |
| Extractor (`facts`, `evidence`) | **waiting for Arckan**: `facts: []`, `evidence: null`, extract stage `not_available` |
| Classifier (labels) | **waiting for Arckan**: labels are the all-true baseline (confidence 0.83), classify stage `not_available` |
| `sample/sample.csv` | arrives upstream 29 Oct 09:00 |

When Arckan's modules land in `app/extract` and `app/classify`, nothing changes in the API shape: `facts`, `evidence`
and real labels start to appear, and `extractor_available` / `classifier_available` turn `true`.

## 8. Integration tests
```powershell
.venv\Scripts\python -m pytest -q tests/runtime/test_api.py     # API: starts, every response matches the schemas
.venv\Scripts\python -m pytest -q tests/runtime                  # whole backend
cd frontend; npm run build                                       # your type-check + build
```
Manual end to end: start the API, `cd frontend; npm run dev`, open http://localhost:5173, press "Use sample.csv":
the banner must not say "API offline", and `GET http://localhost:8000/runs` must list the run.
