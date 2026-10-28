"""Demo API for the frontend (P4). Same pipeline as app/run.py, in memory, on a few rows.

    uvicorn app.api.main:app --reload --port 8000

POST /run           multipart `file` (CSV id,text,date[,label]) or JSON {"use": "sample"} / {"rows": [...]}
GET  /architecture  stage graph {nodes, edges} (app/api/architecture.json, same shape as the frontend mock)
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


ARCHITECTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.json")
STAGE_ORDER = ["load", "extract", "kb", "classify", "generate", "validate", "write"]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/architecture")
def architecture():
    with open(ARCHITECTURE, encoding="utf-8") as f:
        return json.load(f)


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
    """The pipeline on demo rows. Timings are seconds per stage id of the architecture graph."""
    from app.generate import generate

    w = _world()
    lexicon, gold = w["lexicon"], w["gold"]
    timings: dict[str, float] = {}
    details: list[dict] = []
    gold_by_id = {r["id"]: r for r in gold}

    def stage(sid, fn, fallback, status="ok"):
        t = time.time()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out, status = fallback, f"fallback: {e!r}"[:200]
        timings[sid] = round(time.time() - t, 4)
        details.append({"id": sid, "seconds": timings[sid], "status": status})
        return out

    t_load = time.time()
    blank = [dict(r, label="") for r in rows]
    extract, classify = _load_p3()
    timings["load"] = round(time.time() - t_load, 4)
    details.append({"id": "load", "seconds": timings["load"], "status": f"{len(rows)} rows"})
    facts = stage("extract", (lambda: extract(blank, lexicon)) if extract else (lambda: []), [],
                  "ok" if extract else "not available yet (P3): no facts") or []
    # demo rows that come from practice keep their own gold out of the KB, so the demo is honest
    demo_ids = {r["id"] for r in rows}
    kb = stage("kb", lambda: KB(lexicon).build(facts, [g for g in gold if g["id"] not in demo_ids], blank),
               KB(lexicon))
    labels = stage("classify", (lambda: _clean_labels(classify(blank, facts, kb), blank)) if classify else
                   (lambda: None), None, "ok" if classify else "not available yet (P3): baseline true")
    if not labels:
        labels = _clean_labels([], blank)
    report: dict = {}
    texts = w["texts"] | {norm_text(r["text"]) for r in rows}
    claims = stage("generate", lambda: generate(kb, lexicon, texts, 1000, report), [])
    timings["validate"] = round(report.get("validate_s", 0.0), 4)
    timings["generate"] = round(max(0.0, timings["generate"] - timings["validate"]), 4)
    details[-1]["seconds"] = timings["generate"]
    details.append({"id": "validate", "seconds": timings["validate"],
                    "status": f"{report.get('candidates_valid', 0)} valid of {report.get('candidates', 0)} candidates"})
    timings["write"] = 0.0
    details.append({"id": "write", "seconds": 0.0, "status": "in memory (the arena run writes output.csv)"})

    out_labels = []
    for lb in labels:
        g = gold_by_id.get(lb.id)
        out_labels.append({**_plain(lb), "gold": g["label"] if g else None})
    known = [x for x in out_labels if x["gold"] in ("true", "false")]
    runtime = {"rows": len(rows), "total_s": round(sum(timings.values()), 3), "stages": timings,
               "facts": len(facts), "claims": len(claims), "claims_by_type": report.get("by_type", {}),
               "accuracy_vs_gold": round(sum(x["label"] == x["gold"] for x in known) / len(known), 3) if known else None}
    w["last_runtime"] = runtime
    return {"labels": out_labels, "claims": [_plain(c) for c in claims[:claims_limit]],
            "claims_total": len(claims), "claims_by_type": report.get("by_type", {}),
            "facts": [_plain(f) for f in facts], "rows": rows, "timings": timings,
            "stages": STAGE_ORDER, "stage_details": details, "runtime": runtime}


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
