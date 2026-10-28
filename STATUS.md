# STATUS — each member edits only their own row and their own Requests lines

| time | member | done | doing next | blocked by |
|---|---|---|---|---|
| 12:50 | P2 Ioan | metrics from the real pipeline (app/metrics), API v1.1 (runs, claims paging, async, output.csv), docs/API_CONTRACT.md + docs/VALD_HANDOFF.md, DEPENDENCIES.md + CLAUDE.md, deps pinned (constraints.txt, python:3.12.16-slim-trixie); 122 tests + check.sh GREEN | WITH_NEURO build arg + scripts/image_size.sh; wire P3 when merged | — |

## Requests
- @P3 from P2: the runner imports `app.extract.extract(rows, lexicon) -> list[Fact]` and `app.classify.classify(rows, facts, kb) -> list[Label]` lazily and survives their absence or a crash. KB for your features: `kb.lookup(subject, attr, date) -> (value, support, history)`, `app.kb.features.fact_features(kb, subject, attribute, value, date)`, EM hook `kb.timelines()` / `kb.set_reliability({page_id: r})` (page_id = row date).
- @P3 from P2: the KB takes your facts through `app/kb/adapter.py` (Fact dataclass, any object with the same fields, or dicts). A fact whose `value` is not a claimed_value dict (`{"kind": ...}`) is dropped and counted in `kb.stats()['extracted_dropped']`. Extracted facts only veto stale gold values for now; `FactInventory.from_kb(kb, trust_extracted=True)` also trusts values >= 2 rows state on one date - tell me your precision and I switch it on.
- @P1 @P3 from P2: structured output of the sentence classification plugs into claims via `KB.add_classified(labels, facts, rows)` (rows labelled true with confidence >= 0.9 become evidence; on with `TZ_TRUST_CLASSIFIED=1`, off until we measure it). The runner already calls it after `classify`. Each claim keeps its source sentences (`claims_debug.jsonl`, API `claims[].sources`).
- @P4 from P2: read docs/VALD_HANDOFF.md (section 6 is for your agent): new fields predictions/evidence/explanation, claims paging GET /runs/{id}/claims, node status/seconds in /architecture, null = not computed.
- @ALL from P2: DEPENDENCIES.md + CLAUDE.md: read before adding or upgrading any dependency; runtime manifests are P2's.
- @P4 from P2 (earlier): the API is live: `scripts/dev_api.ps1` (or `.sh`) on :8000. `POST /run` returns `{labels, claims, claims_total, claims_by_type, facts, rows, timings (seconds per stage), stages: ["load","extract","kb","classify","generate","validate","write"], stage_details, runtime}`; labels carry `gold` when the row is from practice. `GET /architecture` serves the same graph as your mock (app/api/architecture.json).
- @P1 from P2: `app/io/types.py` holds fallback `Fact/Label/Claim` until `app/contracts.py` lands; it imports yours first. I created this STATUS.md since none existed: restructure freely.
