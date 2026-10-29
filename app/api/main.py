"""Demo API for the frontend (P4): the real pipeline on the rows you send. Contract: docs/API_CONTRACT.md.

    python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000     (scripts/dev_api.ps1 / .sh)

GET  /health                     what is loaded (extractor / classifier available?)
POST /run                        rows in (JSON or CSV upload), labels + claims + metrics out; ?mode=async -> 202;
                                 ?pipeline_mode=prod|debug (debug: half the generated items held out and scored)
GET  /runs                       recent runs (newest first)
GET  /runs/{run_id}              one run's result (status "running" while an async run works)
GET  /runs/{run_id}/claims       the run's 1,000 claims, paged: ?offset=&limit=&type=
GET  /runs/{run_id}/metrics      the run's metrics (schemas.METRICS)
GET  /runs/{run_id}/validation   the claim validator's report: ?verdict=PASS|FAIL|UNVERIFIED&offset=&limit=
GET  /runs/{run_id}/output.csv   the run's table in the official output.csv format
GET  /metrics                    metrics of the latest finished run (+ "offline" from pitch/metrics.json)
GET  /architecture               pipeline graph for React Flow, with each stage's status and seconds
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, is_dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.io.reader import read_lexicon, read_practice
from app.io.writer import COLUMNS, claim_rows, label_rows
from app.kb.features import fact_features
from app.kb.store import KB
from app.metrics import collector
from app.metrics.schemas import STAGES
from app.run import _clean_labels, _load_p3
from app.validate.rules import norm_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRACTICE = os.path.join(ROOT, "data", "practice.csv")
LEXICON = os.path.join(ROOT, "data", "lexicon.json")
SAMPLE = os.path.join(ROOT, "sample", "sample.csv")
OFFLINE_METRICS = os.path.join(ROOT, "pitch", "metrics.json")
STAGE_ORDER = STAGES
MAX_ROWS = 2000
MAX_RUNS = 20
API_VERSION = "1.1"

app = FastAPI(title="Tzatziki · Hype or Truth", version=API_VERSION)
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("TZ_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])

_STATE: dict = {}
_RUNS: "OrderedDict[str, dict]" = OrderedDict()
_LOCK = threading.Lock()


def _world():
    if not _STATE:
        _STATE["gold"] = read_practice(PRACTICE)
        _STATE["lexicon"] = read_lexicon(LEXICON)
        _STATE["texts"] = {norm_text(r["text"]) for r in _STATE["gold"]}
        _STATE["claim_cache"] = {}
    return _STATE


def _practice_facts(extract) -> list[dict]:
    """Facts of every practice row, cross-fitted (each read by models that never saw its page); once per process."""
    w = _world()
    if "practice_facts" not in w:
        rows = [{"id": g["id"], "text": g["text"], "date": g["date"], "label": g["label"]} for g in w["gold"]]
        w["practice_facts"] = extract(rows, w["lexicon"], gold_rows=w["gold"])
    return w["practice_facts"]


def _plain(x):
    if is_dataclass(x):
        d = asdict(x)
        d.pop("meta", None)
        return d
    return dict(x) if isinstance(x, dict) else x


# ---- input ---------------------------------------------------------------------------------------
def _rows_from_csv(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8-sig")
    head = text.split("\n", 1)[0]
    delim = ";" if head.count(";") > head.count(",") else ","
    rows = []
    for r in csv.DictReader(io.StringIO(text), delimiter=delim):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k}
        if r.get("id"):
            rows.append({"id": r["id"], "text": r.get("text", ""), "date": r.get("date", ""),
                         "label": (r.get("label") or "").lower()})
    return rows


def _default_rows(n: int = 20) -> tuple[list[dict], str]:
    if os.path.exists(SAMPLE):
        with open(SAMPLE, "rb") as f:
            return _rows_from_csv(f.read()), "sample/sample.csv"
    gold = _world()["gold"]
    pick = random.Random(int(time.time()) // 60).sample(gold, min(n, len(gold)))
    return [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in pick], "practice (random 20)"


async def _parse_request(request: Request) -> tuple[list[dict], str]:
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "multipart field 'file' missing")
        return _rows_from_csv(await upload.read()), f"upload: {getattr(upload, 'filename', 'file')}"
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - an empty body means "use the sample"
        body = {}
    if isinstance(body, dict) and body.get("rows"):
        return [{"id": str(r.get("id")), "text": str(r.get("text", "")), "date": str(r.get("date", "")), "label": ""}
                for r in body["rows"] if isinstance(r, dict) and r.get("id") is not None], "json rows"
    return _default_rows()


# ---- the pipeline on demo rows -------------------------------------------------------------------
def _evidence(kb: KB, row: dict, facts: list[dict]) -> list[dict]:
    out = []
    for f in facts:
        try:
            v, support, hist = kb.lookup(f["subject"], f["attribute"], row["date"])
            feats = fact_features(kb, f["subject"], f["attribute"], f["value"], row["date"])
        except Exception:  # noqa: BLE001
            continue
        gold_rows = [m.row_id for m in hist if m.source == "gold" and m.date <= row["date"]][-3:]
        out.append({"subject": f["subject"], "attribute": f["attribute"], "stated": f["value"],
                    "value_in_force": v, "agrees": feats["agrees_with_value_in_force"] if v is not None else None,
                    "support": support, "gold_rows": gold_rows,
                    "changed_later": feats["value_changed_after"], "known_wrong_value": feats["known_wrong_value"]})
    return out


def _explain(label, ev: list[dict] | None, classifier: str) -> str:
    if classifier == "baseline":
        return "No classifier yet (P3): the label is the all-true baseline (83% of practice rows are true)."
    if not ev:
        return "No fact of this row is in the knowledge base; the classifier decided from the text alone."
    known = [e for e in ev if e["value_in_force"] is not None]
    bad = [e for e in known if e["agrees"] is False]
    if bad:
        e = bad[0]
        return f"{e['subject']} {e['attribute']}: the value in force on this date differs from the one stated."
    if known:
        return f"{len(known)} of {len(ev)} stated values match the value in force on this date."
    return "The knowledge base has no value in force on this date for these facts."


def run_rows(rows: list[dict], source: str = "rows", claims_limit: int = 50, run_id: str | None = None,
             pipeline_mode: str = "prod") -> dict:
    """The real pipeline on demo rows, as the arena runs it: the practice rows (minus the demo rows) come labelled,
    the demo rows come unlabelled. Demo rows that are practice rows keep their gold out of every KB.
    pipeline_mode: "prod" (train on all generated items) or "debug" (half of them held out and scored)."""
    from app.generate import generate

    w = _world()
    lexicon, gold = w["lexicon"], w["gold"]
    gold_by_id = {r["id"]: r for r in gold}
    run_id = run_id or collector.new_run_id()
    t0 = time.time()
    stages = collector.Stages()

    if pipeline_mode == "debug":
        # debug is measured on the held-out half of the generated sentences: no practice row is hidden (no random
        # 20); only rows that are not practice rows (an uploaded CSV of new sentences) come unlabelled
        rows = [r for r in rows if r["id"] not in gold_by_id]
    blank = [dict(r, label="") for r in rows]
    demo_ids = {r["id"] for r in rows}
    labelled = [{"id": g["id"], "text": g["text"], "date": g["date"], "label": g["label"]}
                for g in gold if g["id"] not in demo_ids]
    universe = labelled + blank
    extract, classify = stages.run("load", _load_p3, (None, None))
    if extract:
        def read():  # practice rows: cached cross-fitted facts; other demo rows: read now
            new = [r for r in blank if r["id"] not in gold_by_id]
            return _practice_facts(extract) + (extract(new, lexicon, gold_rows=gold) if new else [])
        facts_all = stages.run("extract", read, []) or []
    else:
        facts_all = []
        stages.status["extract"] = "not_available"
    facts = [f for f in facts_all if f.get("row_id") in demo_ids]
    kb = stages.run("kb", lambda: KB(lexicon).build(facts, [g for g in gold if g["id"] not in demo_ids], blank),
                    lambda: KB(lexicon))
    labels, classifier, class_report = None, "baseline", {}
    if classify:
        labels = stages.run("classify", lambda: _clean_labels(
            classify(universe, facts_all, kb, mode=pipeline_mode, report=class_report, gold_rows=gold), blank), None)
        # the classifier ran even when no row was left to label (debug: it labels the held-out generated half)
        classifier = "p3" if labels or class_report.get("train_rows") else "baseline"
    else:
        stages.status["classify"] = "not_available"
    details = {}
    if classifier == "p3":
        from app.classify import LAST

        details = LAST.get("details", {})
    if not labels:
        labels = _clean_labels([], blank)

    # Without extracted facts the claims depend only on which gold rows were held out: cache them.
    report: dict = {}
    texts = w["texts"] | {norm_text(r["text"]) for r in rows}
    cache_key = None if facts else (frozenset(demo_ids & set(gold_by_id)), frozenset(norm_text(r["text"]) for r in rows))
    cache = w["claim_cache"]
    if cache_key is not None and cache_key in cache:
        claims, cached = cache[cache_key]
        report.update(cached)
        stages.add("generate", 0.0, "ok")
        stages.add("validate", 0.0, "ok")
        claims_note = "cached: same held-out rows as an earlier run"
    else:
        claims = stages.run("generate", lambda: generate(kb, lexicon, texts, 1000, report), [])
        stages.move("generate", "validate", report.get("validate_s", 0) + report.get("truth_s", 0))
        if cache_key is not None and claims:
            cache.clear()
            cache[cache_key] = (claims, dict(report))
        claims_note = "generated"
    stages.add("write", 0.0, "skipped")  # the demo keeps its table in memory: GET /runs/{id}/output.csv

    facts_plain = [_plain(f) for f in facts]
    by_row: dict[str, list[dict]] = {}
    for f in facts_plain:
        by_row.setdefault(str(f.get("row_id")), []).append(f)
    predictions, out_labels = [], []
    for r, lb in zip(rows, labels):
        g = gold_by_id.get(r["id"])
        rf = by_row.get(r["id"], [])
        ev = _evidence(kb, r, rf) if extract else None
        d = details.get(r["id"], {})
        out_labels.append({**_plain(lb), "gold": g["label"] if g else None, "trust_score": d.get("trust_score")})
        predictions.append({"id": r["id"], "text": r["text"], "date": r["date"], "label": lb.label,
                            "confidence": lb.confidence, "gold": g["label"] if g else None, "facts": rf,
                            "evidence": ev, "explanation": _explain(lb, ev, classifier),
                            "trust_score": d.get("trust_score"), "trust_parts": d.get("trust_parts"),
                            "probabilities": d.get("probabilities"), "sources": d.get("facts", [])})

    held_out_gold = {r["id"]: gold_by_id[r["id"]]["label"] for r in rows if r["id"] in gold_by_id}
    metrics = collector.build(
        run_id=run_id, kind="demo", status=stages.overall(), errors=stages.errors,
        classification=collector.classification_metrics(
            blank, labels, classifier, held_out_gold, "data/practice.csv",
            "demo rows that are practice rows; their gold was held out of the KB for this run"),
        claims=collector.claim_metrics(report, claims),
        runtime=collector.runtime_metrics(stages.timings, stages.status, time.time() - t0, None))
    # a stage that did not run has no time (null), not 0 s
    timings = {s: (round(stages.timings[s], 4) if s in stages.timings and stages.status.get(s) not in
                   ("not_available", "skipped") else None) for s in STAGE_ORDER}
    details = [{"id": s, "seconds": timings[s], "status": stages.status.get(s, "idle")} for s in STAGE_ORDER]
    for d in details:
        if d["id"] == "generate":
            d["note"] = claims_note
    # the classifier's own steps, so the diagram can show them one by one (their seconds sum to "classify")
    shown_stages, timings, details = collector.expand_classify(STAGE_ORDER, timings, details,
                                                               class_report.get("seconds"), pipeline_mode)
    # every claim carries the sentences it was built from (debug; never in output.csv)
    claims_plain = [{**_plain(c), "sources": (getattr(c, "meta", None) or {}).get("sources", [])} for c in claims]
    return {
        "run_id": run_id, "status": stages.overall(), "source": source, "created_at": collector.now_iso(),
        "rows": [{"id": r["id"], "text": r["text"], "date": r["date"]} for r in rows],
        "labels": out_labels, "predictions": predictions, "facts": facts_plain,
        "claims": claims_plain[:claims_limit], "claims_total": len(claims_plain),
        "classifier": {"mode": pipeline_mode, **{k: v for k, v in class_report.items() if k != "generated"},
                       "generated": class_report.get("generated")},
        "claims_by_type": report.get("by_type", {}), "timings": timings, "stages": shown_stages,
        "stage_details": details, "metrics": metrics, "errors": stages.errors,
        "_claims_all": claims_plain, "_labels_obj": labels, "_validation": report.get("_validation_report"),
    }


# ---- run store -----------------------------------------------------------------------------------
def _public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _store(run_id: str, entry: dict) -> None:
    with _LOCK:
        _RUNS[run_id] = entry
        _RUNS.move_to_end(run_id)
        while len(_RUNS) > MAX_RUNS:
            _RUNS.popitem(last=False)


def _get(run_id: str) -> dict:
    with _LOCK:
        e = _RUNS.get(run_id)
    if e is None:
        raise HTTPException(404, f"no run {run_id!r} (the API keeps the last {MAX_RUNS} runs in memory)")
    return e


def _latest_done() -> dict | None:
    with _LOCK:
        for e in reversed(_RUNS.values()):
            if e["status"] != "running" and e.get("result"):
                return e
    return None


def _execute(run_id: str, rows: list[dict], source: str, claims_limit: int, pipeline_mode: str = "prod") -> dict:
    try:
        res = run_rows(rows, source, claims_limit, run_id, pipeline_mode)
        entry = {"run_id": run_id, "status": res["status"], "source": source, "created_at": res["created_at"],
                 "rows": len(rows), "result": res}
    except Exception as e:  # noqa: BLE001 - the stage wrappers make this rare; report it, never 500 silently
        entry = {"run_id": run_id, "status": "failed", "source": source, "created_at": collector.now_iso(),
                 "rows": len(rows), "result": None, "error": repr(e)[:300]}
    _store(run_id, entry)
    return entry


# ---- endpoints -----------------------------------------------------------------------------------
@app.get("/health")
def health():
    extract, classify = _load_p3()
    return {"ok": True, "version": API_VERSION, "extractor_available": extract is not None,
            "classifier_available": classify is not None, "sample_available": os.path.exists(SAMPLE),
            "runs_in_memory": len(_RUNS)}


@app.post("/run")
async def run(request: Request, claims_limit: int = 50, mode: str = "sync", pipeline_mode: str = "prod"):
    if pipeline_mode not in ("prod", "debug"):
        raise HTTPException(400, "pipeline_mode is prod or debug")
    rows, source = await _parse_request(request)
    if not rows:
        raise HTTPException(400, "no rows (need columns id,text,date)")
    if len(rows) > MAX_ROWS:
        raise HTTPException(413, f"the demo endpoint takes at most {MAX_ROWS} rows")
    claims_limit = max(0, min(1000, claims_limit))
    run_id = collector.new_run_id()
    if mode == "async":
        _store(run_id, {"run_id": run_id, "status": "running", "source": source, "created_at": collector.now_iso(),
                        "rows": len(rows), "result": None})
        threading.Thread(target=_execute, args=(run_id, rows, source, claims_limit, pipeline_mode), daemon=True).start()
        return JSONResponse({"run_id": run_id, "status": "running", "poll": f"/runs/{run_id}"}, status_code=202)
    entry = _execute(run_id, rows, source, claims_limit, pipeline_mode)
    if entry["status"] == "failed":
        raise HTTPException(500, {"run_id": run_id, "status": "failed", "error": entry.get("error")})
    return _public(entry["result"])


@app.get("/runs")
def runs():
    with _LOCK:
        return {"runs": [{k: e[k] for k in ("run_id", "status", "source", "created_at", "rows")}
                         for e in reversed(_RUNS.values())]}


@app.get("/runs/{run_id}")
def run_result(run_id: str):
    e = _get(run_id)
    if e["status"] == "running":
        return JSONResponse({"run_id": run_id, "status": "running"}, status_code=202)
    if e["status"] == "failed":
        return JSONResponse({"run_id": run_id, "status": "failed", "error": e.get("error")}, status_code=500)
    return _public(e["result"])


@app.get("/runs/{run_id}/claims")
def run_claims(run_id: str, offset: int = 0, limit: int = 100, type: str | None = None):  # noqa: A002
    e = _get(run_id)
    if not e.get("result"):
        return JSONResponse({"run_id": run_id, "status": e["status"]}, status_code=202 if e["status"] == "running" else 500)
    claims = e["result"]["_claims_all"]
    if type:
        claims = [c for c in claims if c["type"] == type]
    offset, limit = max(0, offset), max(1, min(1000, limit))
    return {"run_id": run_id, "total": len(claims), "offset": offset, "limit": limit,
            "claims": claims[offset:offset + limit]}


@app.get("/runs/{run_id}/validation")
def run_validation(run_id: str, verdict: str | None = None, offset: int = 0, limit: int = 100):
    """The independent validator's report: aggregate counts + one entry per claim (PASS/FAIL/UNVERIFIED)."""
    e = _get(run_id)
    if not e.get("result"):
        return JSONResponse({"run_id": run_id, "status": e["status"]}, status_code=202 if e["status"] == "running" else 500)
    rep = e["result"].get("_validation") or {}
    per = rep.get("claims", [])
    if verdict:
        per = [c for c in per if c["verdict"] == verdict]
    offset, limit = max(0, offset), max(1, min(1000, limit))
    return {"run_id": run_id, "selected": rep.get("selected"), "candidates": rep.get("candidates"),
            "total": len(per), "offset": offset, "limit": limit, "claims": per[offset:offset + limit]}


@app.get("/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    e = _get(run_id)
    if not e.get("result"):
        return JSONResponse({"run_id": run_id, "status": e["status"]}, status_code=202 if e["status"] == "running" else 500)
    return e["result"]["metrics"]


@app.get("/runs/{run_id}/output.csv", response_class=PlainTextResponse)
def run_output_csv(run_id: str):
    e = _get(run_id)
    if not e.get("result"):
        raise HTTPException(409, f"run {run_id} has no result ({e['status']})")
    from app.io.types import Claim

    res = e["result"]
    claims = [Claim(id=c["id"], type=c["type"], text=c["text"], form=c["form"], confidence=c["confidence"])
              for c in res["_claims_all"]]  # sources stay out: the official columns only
    buf = io.StringIO()
    wr = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    wr.writerow(COLUMNS)
    wr.writerows(label_rows(res["_labels_obj"]) + claim_rows(claims))
    return PlainTextResponse(buf.getvalue(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="output-{run_id}.csv"'})


def _offline() -> dict | None:
    if not os.path.exists(OFFLINE_METRICS):
        return None
    try:
        with open(OFFLINE_METRICS, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


@app.get("/metrics")
def metrics():
    e = _latest_done()
    m = json.loads(json.dumps(e["result"]["metrics"])) if e else collector.empty()
    offline = _offline()
    m["offline"] = {"source": "pitch/metrics.json (measured outside this API: P3's model evaluation, P2's runtime runs)",
                    "data": offline}
    if m["model"] is None and offline and isinstance(offline.get("model"), dict):
        m["model"] = {**offline["model"], "label_source": "offline evaluation (pitch/metrics.json)"}
    if offline and "neuro" in offline:
        m["neuro"] = offline["neuro"]
    return m


@app.get("/architecture")
def architecture():
    e = _latest_done()
    if not e:
        return collector.architecture()
    res = e["result"]
    return collector.architecture(res["metrics"]["runtime"]["stage_status"], res["timings"], res["run_id"])


# Training CSV vs validation CSV (POST /training, POST /validate): app/api/validation.py
from app.api.validation import router as _validation_router  # noqa: E402

app.include_router(_validation_router)
