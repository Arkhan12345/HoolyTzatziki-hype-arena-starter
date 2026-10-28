# STATUS — each member edits only their own row and their own Requests lines

| time | member | done | doing next | blocked by |
|---|---|---|---|---|
| 11:40 | P2 Ioan | Dockerfile, app/run.py (fallback, watchdog, SIGTERM), app/io, app/kb, app/generate, app/validate, app/api; tests/runtime green; scripts/local_check.py | wire P3's extract/classify when merged; check.sh on a machine with Docker | Docker not installed on Ioan's laptop |

## Requests
- @P3 from P2: the runner imports `app.extract.extract(rows, lexicon) -> list[Fact]` and `app.classify.classify(rows, facts, kb) -> list[Label]` lazily and survives their absence or a crash. KB for your features: `kb.lookup(subject, attr, date) -> (value, support, history)`, `app.kb.features.fact_features(kb, subject, attribute, value, date)`, EM hook `kb.timelines()` / `kb.set_reliability({page_id: r})` (page_id = row date).
- @P4 from P2: the API is live: `scripts/dev_api.ps1` (or `.sh`) on :8000. `POST /run` returns `{labels, claims, claims_total, claims_by_type, facts, rows, timings (seconds per stage), stages: ["load","extract","kb","classify","generate","validate","write"], stage_details, runtime}`; labels carry `gold` when the row is from practice. `GET /architecture` serves the same graph as your mock (app/api/architecture.json).
- @P1 from P2: `app/io/types.py` holds fallback `Fact/Label/Claim` until `app/contracts.py` lands; it imports yours first. I created this STATUS.md since none existed: restructure freely.
