---
name: arena-runtime
description: Use when changing or verifying Team Tzatziki's runtime (P2/Ioan) - Dockerfile, app/run.py, app/io, app/kb, app/generate, app/validate, app/api - or before pushing anything that could break /output/output.csv or the 1,000 claims.
---

# Arena runtime: change it, then prove it still hands in a valid table

The run QA DNA executes is `python -m app.run` inside the Docker image. It must ALWAYS leave a valid
`/output/output.csv` (labels for every blank row + exactly 1,000 true, new, validator-clean claims).
Rules live in `contracts/` (never edit it): `README.md`, `claim-forms.md`, `value-forms.md`, the two schemas.

## Map
| Path | Job |
| --- | --- |
| `app/run.py` | pipeline, fallback table after the gold claims, watchdog at deadline-90 s, SIGTERM write |
| `app/io/` | CSV reader (BOM, `;` or `,`), atomic writer (`.output.*.tmp` then `os.replace`), fallback types |
| `app/kb/` | `KB` timelines per (subject, attribute): gold (practice true), wrong (practice false, 1 fact), extracted (P3) |
| `app/generate/` | `render.py` value forms, `templates.py` one sentence per type, `select.py` candidates + quotas |
| `app/validate/rules.py` | schema + lexicon + claim-forms text rules; conservative (rejects what the referee might misread) |
| `app/api/main.py` | demo API for the frontend: `POST /run`, `GET /architecture`, `GET /metrics` |
| `tests/runtime/` | example tables from `claim-forms.md` as pytest cases, generator invariants, run/SIGTERM/crash tests |

P3 (Arckan) plugs in through `app.extract.extract(rows, lexicon)` and `app.classify.classify(rows, facts, kb)`;
the runner imports them lazily and survives their absence or a crash.

## Verify (always, in this order)
```sh
.venv/Scripts/python -m pytest -q tests/runtime          # ~20 s; must be green
.venv/Scripts/python scripts/local_check.py --minutes 5   # check.sh without Docker + validator over every claim
.venv/Scripts/python scripts/synth_universe.py "$TEMP/u20k" && .venv/Scripts/python scripts/local_check.py --universe "$TEMP/u20k"
sh scripts/check.sh                                       # the real thing; needs Docker (not on Ioan's laptop yet)
```
`local_check.py` cannot test user 10001, the read-only root, the memory cap or the missing network: only
`check.sh` can. Run it before the T3 freeze on a machine with Docker.

## Rules when editing the generator
- Truth comes from gold only. A `change` pairs two gold values at increasing dates (safest type). A `fact`
  needs a recent last mention (Sep 2026 on ~96% right in subsampling, earlier is much worse).
- Every new template: add its example to `tests/runtime/test_validate.py` and keep `test_exactly_1000_unique_valid` green.
- Values are written by `render.value_text` only. Never hand-format a number: `2,026 customers` (a bare 2026 is a year),
  no 3-decimal numbers outside percent, money symbols for EUR/USD/GBP else the ISO code after the amount.
- A compare sentence holds no number, date or time word; amounts use `higher/lower than`, `the same as`; dates `after/before`.

## Team rules that bind this session
Work on branch `john` until prod. Commit message `P2-backend: <what>`. Only edit P2 paths (see `team/Ioan_backend+devops.md`);
needs from others go to `STATUS.md` → Requests. One line per meaningful commit in `pitch/JOURNAL.md`.
