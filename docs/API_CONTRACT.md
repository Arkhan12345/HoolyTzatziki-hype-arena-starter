# Demo API contract (P2 → P4)

Implemented in `app/api/main.py`. Machine-readable schemas: `app/metrics/schemas.py` (`RUN_RESULT`, `METRICS`,
`ARCHITECTURE`); `tests/runtime/test_api.py` validates every response against them. API version `1.1`.

Base URL: `http://localhost:8000` (frontend env `VITE_API_URL`). CORS: `http://localhost:5173`, `http://127.0.0.1:5173`
(override with `TZ_CORS_ORIGINS=origin1,origin2`).

**Null means "not computed", never 0.** A stage that did not run has `null` seconds; an evaluation with no known
labels is `null`.

## Endpoints

| Method | Path | Returns | Status codes |
| --- | --- | --- | --- |
| GET | `/health` | `{ok, version, extractor_available, classifier_available, sample_available, runs_in_memory}` | 200 |
| POST | `/run` | `RunResult` (sync) or `{run_id, status: "running", poll}` with `?mode=async` | 200, 202, 400, 413, 500 |
| GET | `/runs` | `{runs: [{run_id, status, source, created_at, rows}]}`, newest first, last 20 kept in memory | 200 |
| GET | `/runs/{run_id}` | `RunResult`; `{run_id, status: "running"}` while an async run works | 200, 202, 404, 500 |
| GET | `/runs/{run_id}/claims?offset=0&limit=100&type=compare` | `{run_id, total, offset, limit, claims: [Claim]}`; `limit` ≤ 1000 | 200, 202, 404 |
| GET | `/runs/{run_id}/metrics` | `Metrics` of that run | 200, 202, 404 |
| GET | `/runs/{run_id}/output.csv` | the run's table in the official `output.csv` format (`text/csv`) | 200, 404, 409 |
| GET | `/metrics` | `Metrics` of the latest finished run (all-null if none) + `offline` + optional `neuro` | 200 |
| GET | `/architecture` | `Architecture` with each node's `status` and `seconds` from the latest run | 200 |

### POST /run input
One of:
- `Content-Type: application/json`, body `{"use": "sample"}` (or empty): `sample/sample.csv` if it exists, else 20 random practice rows.
- `Content-Type: application/json`, body `{"rows": [{"id": "x-1", "text": "…", "date": "2026-10-01"}]}`.
- `multipart/form-data`, field `file`: a CSV with columns `id,text,date[,label]`, `,` or `;` separated, BOM ok.

Query: `claims_limit` (0–1000, default 50: how many claims ride along; all 1,000 via `/runs/{id}/claims`),
`mode=sync|async` (default `sync`). At most 2,000 rows.

## Types

```ts
type Verdict = "true" | "false" | "unsure"
type Stage = "load" | "extract" | "kb" | "classify" | "generate" | "validate" | "write"
type StageStatus = "idle" | "ok" | "fallback" | "not_available" | "failed" | "skipped"

interface Label { id: string; label: Verdict; confidence: number | null; gold: Verdict | null }   // gold only for practice rows
interface Claim { id: string; type: "fact"|"compare"|"change"|"at_event"|"relation"; text: string; form: object; confidence: number }
interface Fact  { row_id: string; subject: string; attribute: string; value: {kind: string, ...}; span: string; conf: number }
interface Evidence {                                  // one per extracted fact of the row
  subject: string; attribute: string; stated: object
  value_in_force: object | null                       // the KB's value on the row's date
  agrees: boolean | null; support: number; gold_rows: string[]
  changed_later: boolean; known_wrong_value: boolean
}
interface Prediction {
  id: string; text: string; date: string; label: Verdict; confidence: number | null; gold: Verdict | null
  facts: Fact[]                                        // [] until P3's extractor is merged
  evidence: Evidence[] | null                          // null when there is no extractor
  explanation: string                                  // one sentence, generated from the above
}
interface RunResult {
  run_id: string; status: "ok" | "partial" | "failed"; source: string; created_at: string
  rows: {id: string; text: string; date: string}[]
  labels: Label[]; predictions: Prediction[]; facts: Fact[]
  claims: Claim[]                                      // first `claims_limit`
  claims_total: number; claims_by_type: Record<string, number>
  timings: Record<Stage, number | null>                // seconds
  stages: Stage[]                                      // always the 7 stages in order
  stage_details: {id: Stage; seconds: number | null; status: StageStatus; note?: string}[]
  metrics: Metrics; errors: {stage: string; message: string}[]
}
interface Metrics {
  schema_version: "1.0"; run_id: string | null; kind: "arena" | "demo" | "none"; created_at: string | null
  status: "ok" | "partial" | "failed" | "running" | "none"; errors: {stage: string; message: string}[]
  classification: {
    rows_total: number; rows_labelled_in_input: number; rows_predicted: number
    predicted: {true: number; false: number; unsure: number}
    confidence: ConfidenceStats | null; classifier: "p3" | "baseline"
    evaluation: null | {                               // only over rows whose label is known from data/practice.csv
      label_source: string; note: string; n: number; accuracy: number | null; macro_f1: number | null
      baseline: {name: "all true"; accuracy: number | null; macro_f1: number | null}
      confusion: {labels: ["true","false","unsure"]; matrix: number[][]}   // rows = gold, columns = predicted
      calibration: {predicted: number; observed: number; count: number}[]; ece: number | null
    }
  }
  claims: {
    candidates_total: number|null; candidates_by_type: Record<string, number>
    rejected_by_reason: Record<string, Record<string, number>>   // type -> reason -> count
    valid: number|null; unique_text: number|null; unique_semantic: number|null; semantic_duplicates: number|null
    selected: number|null; selected_by_type: Record<string, number>; selected_semantic_duplicates: number|null
    confidence: ConfidenceStats | null; gold_only_supported: number|null; kb_supported: number|null
    inventory: object | null
  }
  runtime: { total_s: number|null; stages: Record<Stage, number|null>; stage_status: Record<Stage, StageStatus>; rss_mb: number|null }
  model: object | null          // = classification.evaluation (kept for the /metrics page); /metrics may fill it from offline P3 numbers, tagged label_source
  claims_by_type: Record<string, number>
  offline?: {source: string; data: object | null}      // GET /metrics only: pitch/metrics.json as is
  neuro?: object                                       // GET /metrics only, when pitch/metrics.json has it
}
interface ConfidenceStats { n: number; min: number; max: number; mean: number; median: number; histogram: {lo: number; hi: number; count: number}[] }
interface Architecture {
  nodes: {id: string; label: string; kind: "io"|"stage"|"optional"; stage: Stage; owner: string; description: string
          details?: string[]; position?: {x: number; y: number}; status: StageStatus; seconds: number | null}[]
  edges: {source: string; target: string; label?: string; dashed?: boolean; loop?: boolean}[]
  stage_status: Record<Stage, StageStatus>; timings: Record<Stage, number | null>; run_id: string | null
}
```

## Files the pipeline writes

| File | Written by | Where | Official? |
| --- | --- | --- | --- |
| `output.csv` | `python -m app.run` | `$OUTPUT_DIR` (arena: `/output/output.csv`) | **yes**, the only one QA DNA reads |
| `metrics.json` (`Metrics`, `kind: "arena"`) | `python -m app.run`, after `output.csv` | `$TZ_AUX_DIR`; default: `/tmp/tzatziki` when `OUTPUT_DIR=/output`, else next to `output.csv` | no |
| `architecture.json` (`Architecture`) | same | same | no |

The arena's `/output` only ever holds `output.csv`. A failure writing the auxiliary files is logged and ignored.
