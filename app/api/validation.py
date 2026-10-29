"""Training CSV vs validation CSV, for the frontend's Validate page.

- POST /training   a labelled CSV (id, text, date, label, optional form). Its rows join practice.csv as truth:
                   they pin the knowledge base and the classifier trains on them. A row without a `form` gets the
                   facts the extractor reads from it. Replaces the previous upload; DELETE /training clears it.
- POST /validate   a CSV of statements to judge. It only runs extract -> KB -> classify (no claims, no output.csv).
                   A `label` column, if filled, is the expected answer: it is kept out of the run and only used to
                   mark each statement right or wrong.

Every statement comes back with how it was judged: the facts read from it, the value in force on its date, the
rows that support or contradict it, the class probabilities and a one-line explanation.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time

from fastapi import APIRouter, HTTPException, Request

from app.api import main as M

router = APIRouter()
LABELS = ("true", "false", "unsure")
_TRAINING: dict = {"rows": [], "facts": [], "source": None, "summary": None}
_RUN_LOCK = threading.Lock()  # one classifier run at a time: it keeps its details in a module global


def _read_csv(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8-sig")
    head = text.split("\n", 1)[0]
    delim = ";" if head.count(";") > head.count(",") else ","
    out = []
    for r in csv.DictReader(io.StringIO(text), delimiter=delim):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k}
        if r.get("id"):
            out.append(r)
    return out


async def _upload(request: Request) -> tuple[bytes, str]:
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise HTTPException(400, "send the CSV as multipart field 'file'")
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        raise HTTPException(400, "multipart field 'file' missing")
    return await upload.read(), getattr(upload, "filename", "file")


def _form(raw: str) -> list[dict]:
    try:
        form = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [f for f in form if isinstance(f, dict) and f.get("subject") and f.get("attribute") and "value" in f] \
        if isinstance(form, list) else []


def _training_summary() -> dict:
    s = _TRAINING["summary"]
    practice = len(M._world()["gold"])
    return {"practice_rows": practice, "uploaded": s, "total_labelled": practice + len(_TRAINING["rows"])}


# ---- training CSV --------------------------------------------------------------------------------
@router.post("/training")
async def upload_training(request: Request):
    raw, name = await _upload(request)
    w = M._world()
    practice_ids = {g["id"] for g in w["gold"]}
    rows, skipped = [], []
    for r in _read_csv(raw):
        label = (r.get("label") or "").lower()
        if label not in LABELS:
            skipped.append({"id": r["id"], "reason": f"label {label or '(empty)'!r} is not true, false or unsure"})
        elif r["id"] in practice_ids:
            skipped.append({"id": r["id"], "reason": "id already in practice.csv (practice keeps its label)"})
        elif not r.get("text") or not r.get("date"):
            skipped.append({"id": r["id"], "reason": "text or date missing"})
        else:
            rows.append({"id": r["id"], "text": r["text"], "date": r["date"][:10], "label": label,
                         "form": _form(r.get("form", ""))})
    if not rows:
        raise HTTPException(400, {"message": "no usable labelled row", "skipped": skipped[:50]})

    # Rows without a form: the facts the extractor reads stand in for it.
    extract, _ = M._load_p3()
    need = [r for r in rows if not r["form"]]
    read = extract([dict(r, label="") for r in need], w["lexicon"], gold_rows=w["gold"]) if (extract and need) else []
    by_row: dict[str, list[dict]] = {}
    for f in read:
        by_row.setdefault(str(f["row_id"]), []).append(f)
    facts = []
    for r in rows:
        if r["form"]:
            facts += [{"row_id": r["id"], "subject": f["subject"], "attribute": f["attribute"], "value": f["value"],
                       "conf": 1.0, "span": "", "anchored": True} for f in r["form"]]
        else:
            got = by_row.get(r["id"], [])
            r["form"] = [{"subject": f["subject"], "attribute": f["attribute"], "value": f["value"]} for f in got]
            facts += got

    counts = {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}
    _TRAINING.update(rows=rows, facts=facts, source=name, summary={
        "file": name, "rows": len(rows), "labels": counts,
        "with_form": len(rows) - len(need), "facts_read_by_extractor": sum(bool(r["form"]) for r in need),
        "without_any_fact": sum(not r["form"] for r in rows), "skipped": skipped[:50], "skipped_total": len(skipped),
        "uploaded_at": M.collector.now_iso()})
    return _training_summary()


@router.get("/training")
def get_training():
    return _training_summary()


@router.delete("/training")
def clear_training():
    _TRAINING.update(rows=[], facts=[], source=None, summary=None)
    return _training_summary()


# ---- validation CSV ------------------------------------------------------------------------------
def validate_rows(rows: list[dict], source: str) -> dict:
    """Judge `rows` with practice.csv + the uploaded training rows as the labelled truth. No claims."""
    w = M._world()
    lexicon = w["lexicon"]
    ids = {r["id"] for r in rows}
    truth = [g for g in w["gold"] + _TRAINING["rows"] if g["id"] not in ids]
    practice_ids = {g["id"] for g in w["gold"]}
    blank = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]
    universe = [{"id": g["id"], "text": g["text"], "date": g["date"], "label": g["label"]} for g in truth] + blank
    timings: dict[str, float] = {}

    t = time.time()
    extract, classify = M._load_p3()
    if not extract or not classify:
        raise HTTPException(503, "the extractor or classifier is not available in this build")
    timings["load"] = time.time() - t

    t = time.time()
    new = [r for r in blank if r["id"] not in practice_ids]
    facts_all = list(M._practice_facts(extract))  # practice rows, cross-fitted (also when one is being judged)
    facts_all +=[f for f in _TRAINING["facts"] if f["row_id"] not in ids]
    facts_all += extract(new, lexicon, gold_rows=w["gold"]) if new else []
    facts = [f for f in facts_all if f.get("row_id") in ids]
    timings["extract"] = time.time() - t

    t = time.time()
    kb = M.KB(lexicon).build(facts, truth, blank)
    timings["kb"] = time.time() - t

    t = time.time()
    report: dict = {}
    with _RUN_LOCK:
        labels = M._clean_labels(classify(universe, facts_all, kb, mode="prod", report=report, gold_rows=truth), blank)
        from app.classify import LAST

        details = dict(LAST.get("details", {}))
    timings["classify"] = time.time() - t

    facts_plain = [M._plain(f) for f in facts]
    by_row: dict[str, list[dict]] = {}
    for f in facts_plain:
        by_row.setdefault(str(f.get("row_id")), []).append(f)
    statements = []
    for r, lb in zip(rows, labels):
        rf = by_row.get(r["id"], [])
        ev = M._evidence(kb, r, rf)
        d = details.get(r["id"], {})
        expected = r.get("expected") or None
        statements.append({
            "id": r["id"], "text": r["text"], "date": r["date"], "label": lb.label, "confidence": lb.confidence,
            "expected": expected, "correct": (lb.label == expected) if expected else None,
            "facts": rf, "evidence": ev, "explanation": M._explain(lb, ev, "p3"),
            "probabilities": d.get("probabilities"), "trust_score": d.get("trust_score"),
            "trust_parts": d.get("trust_parts"), "sources": d.get("facts", [])})

    scored = [s for s in statements if s["expected"]]
    return {
        "source": source, "created_at": M.collector.now_iso(),
        "training": _training_summary(),
        "summary": {
            "statements": len(statements),
            "predicted": {lab: sum(s["label"] == lab for s in statements) for lab in LABELS},
            "with_facts": sum(bool(s["facts"]) for s in statements),
            "expected_known": len(scored),
            "correct": sum(bool(s["correct"]) for s in scored),
            "accuracy": (sum(bool(s["correct"]) for s in scored) / len(scored)) if scored else None,
        },
        "classifier": {k: v for k, v in report.items() if k not in ("generated", "debug")},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "statements": statements,
    }


@router.post("/validate")
async def validate(request: Request):
    raw, name = await _upload(request)
    rows = []
    for r in _read_csv(raw):
        expected = (r.get("label") or "").lower()
        rows.append({"id": r["id"], "text": r.get("text", ""), "date": (r.get("date") or "")[:10],
                     "expected": expected if expected in LABELS else ""})
    rows = [r for r in rows if r["text"] and r["date"]]
    if not rows:
        raise HTTPException(400, "no rows (need columns id,text,date)")
    if len(rows) > M.MAX_ROWS:
        raise HTTPException(413, f"the demo endpoint takes at most {M.MAX_ROWS} rows")
    return validate_rows(rows, f"upload: {name}")
