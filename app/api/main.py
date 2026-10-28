"""Demo API for the frontend (P4). Same pipeline as app/run.py, in memory, on a few rows.

    uvicorn app.api.main:app --reload --port 8000

POST /run           multipart `file` (CSV id,text,date[,label]) or JSON {"use": "sample"} / {"rows": [...]}
GET  /architecture  stage graph {nodes, edges}
GET  /metrics       pitch/metrics.json (+ "runtime" from the last /run)
GET  /health
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.io.reader import read_lexicon, read_practice
from app.kb.store import KB
from app.run import BASELINE_TRUE_CONF, _clean_labels, _load_p3
from app.validate.rules import norm_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRACTICE = os.path.join(ROOT, "data", "practice.csv")
LEXICON = os.path.join(ROOT, "data", "lexicon.json")
SAMPLE = os.path.join(ROOT, "sample", "sample.csv")
METRICS = os.path.join(ROOT, "pitch", "metrics.json")

app = FastAPI(title="Tzatziki · Hype or Truth", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_STATE: dict = {}


def _world():
    if not _STATE:
        _STATE["gold"] = read_practice(PRACTICE)
        _STATE["lexicon"] = read_lexicon(LEXICON)
        _STATE["texts"] = {norm_text(r["text"]) for r in _STATE["gold"]}
        _STATE["last_runtime"] = {}
    return _STATE


STAGES = [
    {"id": "input", "label": "universe.csv + lexicon.json", "owner": "QA DNA",
     "description": "Rows id, text, date, label; the lexicon of subjects, aliases, attributes and events."},
    {"id": "extract", "label": "Extractor", "owner": "Arckan",
     "description": "Lexicon gazetteer, value regexes per value-forms.md, attribute classifier, page coreference."},
    {"id": "kb", "label": "Temporal knowledge base", "owner": "Ioan",
     "description": "Timeline per subject + attribute from practice gold and extracted facts; support votes; page trust."},
    {"id": "classify", "label": "Truth classifier", "owner": "Arckan",
     "description": "Features from the KB (agrees with the value in force, contradicts it, support, page trust) into LightGBM, calibrated."},
    {"id": "generate", "label": "Claim generator", "owner": "Ioan",
     "description": "Templates for fact, compare, change, at_event and relation, only from firmly supported facts."},
    {"id": "validate", "label": "Validator", "owner": "Ioan",
     "description": "entry.schema.json, lexicon names and attributes, claim-forms.md sentence rules, novelty."},
    {"id": "write", "label": "Writer", "owner": "Ioan",
     "description": "Fallback table within seconds, watchdog and SIGTERM writes, temp file then rename."},
    {"id": "output", "label": "/output/output.csv", "owner": "QA DNA",
     "description": "One label per unlabelled row and exactly 1,000 new true claims."},
]
EDGES = [
    {"from": "input", "to": "extract"}, {"from": "extract", "to": "kb"}, {"from": "kb", "to": "classify"},
    {"from": "kb", "to": "generate"}, {"from": "generate", "to": "validate"},
    {"from": "validate", "to": "generate", "label": "invalid → next candidate"},
    {"from": "classify", "to": "write", "label": "labels"}, {"from": "validate", "to": "write", "label": "1,000 claims"},
    {"from": "write", "to": "output"},
]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/architecture")
def architecture():
    return {"nodes": STAGES, "edges": EDGES}


@app.get("/metrics")
def metrics():
    data = {}
    if os.path.exists(METRICS):
        with open(METRICS, encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault("runtime", {})
    data["runtime"] = {**data["runtime"], "last_demo_run": _world()["last_runtime"]}
    return data


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


def _plain(x):
    if is_dataclass(x):
        d = asdict(x)
        d.pop("meta", None)
        return d
    return x


def run_rows(rows: list[dict], claims_limit: int = 50) -> dict:
    from app.generate import generate

    w = _world()
    lexicon, gold = w["lexicon"], w["gold"]
    stages, timings = [], {}
    gold_by_id = {r["id"]: r for r in gold}

    def stage(sid, fn, fallback):
        t = time.time()
        status = "ok"
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out, status = fallback, f"fallback: {e!r}"[:200]
        timings[sid] = round((time.time() - t) * 1000, 1)
        stages.append({"id": sid, "ms": timings[sid], "status": status})
        return out

    extract, classify = _load_p3()
    facts = stage("extract", (lambda: extract(rows, lexicon)) if extract else (lambda: []), []) or []
    if not extract:
        stages[-1]["status"] = "not available (P3)"
    # demo rows from practice keep their gold out of the KB, so the demo is honest
    demo_ids = {r["id"] for r in rows}
    kb = stage("kb", lambda: KB(lexicon).build(facts, [g for g in gold if g["id"] not in demo_ids], rows),
               KB(lexicon))
    blank = [dict(r, label="") for r in rows]
    if classify:
        labels = stage("classify", lambda: _clean_labels(classify(blank, facts, kb), blank), None)
    else:
        labels = None
        stage("classify", lambda: None, None)
        stages[-1]["status"] = "not available (P3): baseline true"
    if not labels:
        labels = _clean_labels([], blank)
    report: dict = {}
    texts = w["texts"] | {norm_text(r["text"]) for r in rows}
    claims = stage("generate", lambda: generate(kb, lexicon, texts, 1000, report), [])
    stages.append({"id": "validate", "ms": 0, "status": f"{report.get('candidates_valid', 0)} valid candidates"})

    out_labels = []
    for lb in labels:
        g = gold_by_id.get(lb.id)
        out_labels.append({**_plain(lb), "gold": g["label"] if g else None})
    known = [x for x in out_labels if x["gold"] in ("true", "false")]
    runtime = {"rows": len(rows), "timings_ms": timings, "facts": len(facts), "claims": len(claims),
               "claims_by_type": report.get("by_type", {}),
               "accuracy_vs_gold": round(sum(x["label"] == x["gold"] for x in known) / len(known), 3) if known else None,
               "baseline_true_conf": BASELINE_TRUE_CONF}
    w["last_runtime"] = runtime
    return {"labels": out_labels, "claims": [_plain(c) for c in claims[:claims_limit]],
            "claims_total": len(claims), "facts": [_plain(f) for f in facts],
            "timings": timings, "stages": stages, "runtime": runtime}


@app.post("/run")
async def run(request: Request, claims_limit: int = 50):
    ctype = request.headers.get("content-type", "")
    source = "upload"
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "multipart field 'file' missing")
        rows = _rows_from_csv(await upload.read())
    else:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if isinstance(body, dict) and body.get("rows"):
            rows, source = [{"id": str(r.get("id")), "text": r.get("text", ""), "date": r.get("date", ""),
                             "label": ""} for r in body["rows"]], "json"
        else:
            rows, source = _default_rows()
    if not rows:
        raise HTTPException(400, "no rows (need columns id,text,date)")
    if len(rows) > 2000:
        raise HTTPException(413, "demo endpoint takes at most 2,000 rows")
    return {"source": source, **run_rows(rows, claims_limit)}
