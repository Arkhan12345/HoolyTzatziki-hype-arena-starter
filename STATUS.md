# STATUS — each member edits only their own row and their own Requests lines

| time | member | done | doing next | blocked by |
|---|---|---|---|---|
| 12:10 | P2 Ioan | generator v2: FactInventory contract, no type quotas, confidence-first, both compare orientations, truth re-check, capacity report; check.sh GREEN in Docker (WSL): exit 0, RSS 43 MB, image 689 MB, 19k rows in 4 s; Dockerfile, app/run.py (fallback, watchdog, SIGTERM), app/io, app/kb, app/generate, app/validate, app/api; tests/runtime green; scripts/local_check.py | wire P3's extract/classify when merged; WITH_NEURO build arg + scripts/image_size.sh | — |

## Requests
- @P3 from P2: the runner imports `app.extract.extract(rows, lexicon) -> list[Fact]` and `app.classify.classify(rows, facts, kb) -> list[Label]` lazily and survives their absence or a crash. KB for your features: `kb.lookup(subject, attr, date) -> (value, support, history)`, `app.kb.features.fact_features(kb, subject, attribute, value, date)`, EM hook `kb.timelines()` / `kb.set_reliability({page_id: r})` (page_id = row date).
- @P3 from P2: the KB takes your facts through `app/kb/adapter.py` (Fact dataclass, any object with the same fields, or dicts). A fact whose `value` is not a claimed_value dict (`{"kind": ...}`) is dropped and counted in `kb.stats()['extracted_dropped']`. Extracted facts only veto stale gold values for now; `FactInventory.from_kb(kb, trust_extracted=True)` also trusts values >= 2 rows state on one date - tell me your precision and I switch it on.
- @P4 from P2: the API is live: `scripts/dev_api.ps1` (or `.sh`) on :8000. `POST /run` returns `{labels, claims, claims_total, claims_by_type, facts, rows, timings (seconds per stage), stages: ["load","extract","kb","classify","generate","validate","write"], stage_details, runtime}`; labels carry `gold` when the row is from practice. `GET /architecture` serves the same graph as your mock (app/api/architecture.json).
- @P1 from P2: `app/io/types.py` holds fallback `Fact/Label/Claim` until `app/contracts.py` lands; it imports yours first. I created this STATUS.md since none existed: restructure freely.
