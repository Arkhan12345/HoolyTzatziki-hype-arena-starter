import csv
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import jsonschema
import pytest
from fastapi.testclient import TestClient

from app.api.main import STAGE_ORDER, app
from app.metrics.schemas import ARCHITECTURE, METRICS, RUN_RESULT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
client = TestClient(app)


@pytest.fixture(scope="module")
def sample_run():
    r = client.post("/run", json={"use": "sample"})
    assert r.status_code == 200, r.text
    return r.json()


def test_api_starts_with_uvicorn():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                    body = json.load(resp)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("API did not answer /health within 20 s")
        assert body["ok"] is True and "extractor_available" in body
    finally:
        p.terminate()
        p.wait(timeout=10)


def test_run_result_matches_contract(sample_run):
    jsonschema.validate(sample_run, RUN_RESULT)
    assert sample_run["stages"] == STAGE_ORDER
    assert len(sample_run["labels"]) == len(sample_run["rows"]) == len(sample_run["predictions"]) > 0
    assert sample_run["claims_total"] == 1000 and len(sample_run["claims"]) == 50


def test_frontend_payload_is_stable(sample_run):
    """Keys frontend/src/api/types.ts (RunResponse) reads must not move."""
    for key in ("labels", "claims", "facts", "timings", "stages", "rows"):
        assert key in sample_run
    lb = sample_run["labels"][0]
    assert set(lb) >= {"id", "label", "confidence"}
    cl = sample_run["claims"][0]
    assert set(cl) >= {"id", "type", "text", "form", "confidence", "sources"} and "meta" not in cl
    assert cl["sources"] and {"row_id", "date", "text"} <= set(cl["sources"][0])
    assert set(sample_run["rows"][0]) == {"id", "text", "date"}
    assert all(v is None or isinstance(v, (int, float)) for v in sample_run["timings"].values())


def test_unavailable_numbers_are_null_not_invented():
    fresh = TestClient(app)
    m = fresh.get("/metrics").json()
    jsonschema.validate(m, METRICS)
    # without a classifier the confidence is the baseline's; evaluation only over rows with known gold
    r = fresh.post("/run", json={"rows": [{"id": "zz-1", "text": "Hooly AI Pro costs €59 per month.", "date": "2026-10-01"}]}).json()
    assert r["metrics"]["classification"]["evaluation"] is None  # zz-1 has no known label
    assert r["metrics"]["model"] is None
    assert r["predictions"][0]["gold"] is None


def test_metrics_after_a_run(sample_run):
    m = client.get("/metrics").json()
    jsonschema.validate(m, METRICS)
    latest = client.get("/runs").json()["runs"][0]["run_id"]  # other tests may have run since
    assert m["run_id"] == latest and m["claims"]["selected"] == 1000
    assert m["claims_by_type"] == client.get(f"/runs/{latest}").json()["claims_by_type"]
    assert "offline" in m
    rm = client.get(f"/runs/{sample_run['run_id']}/metrics").json()
    jsonschema.validate(rm, METRICS)


def test_architecture_for_react_flow(sample_run):
    a = client.get("/architecture").json()
    jsonschema.validate(a, ARCHITECTURE)
    ids = {n["id"] for n in a["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in a["edges"])
    assert a["run_id"] == client.get("/metrics").json()["run_id"]


def test_claims_paging_and_filter(sample_run):
    rid = sample_run["run_id"]
    page = client.get(f"/runs/{rid}/claims?offset=0&limit=1000").json()
    assert page["total"] == 1000 and len(page["claims"]) == 1000
    comp = client.get(f"/runs/{rid}/claims?type=compare&limit=5").json()
    assert all(c["type"] == "compare" for c in comp["claims"])
    assert client.get("/runs/does-not-exist/claims").status_code == 404


def test_output_csv_is_the_official_format(sample_run):
    text = client.get(f"/runs/{sample_run['run_id']}/output.csv").text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0]) == ["id", "type", "text", "form", "label", "confidence"]
    claims = [r for r in rows if r["type"]]
    labels = [r for r in rows if not r["type"]]
    assert len(claims) == 1000 and len(labels) == len(sample_run["rows"])
    assert all(isinstance(json.loads(r["form"]), dict) for r in claims)


def test_upload_semicolon_csv():
    data = "id;text;date;label\nx-1;Hooly AI Pro costs €59 per month.;2026-10-01;\n".encode("utf-8-sig")
    r = client.post("/run", files={"file": ("s.csv", io.BytesIO(data), "text/csv")}).json()
    assert [lb["id"] for lb in r["labels"]] == ["x-1"]


def test_async_run_then_poll():
    r = client.post("/run?mode=async", json={"use": "sample"})
    assert r.status_code == 202
    rid = r.json()["run_id"]
    for _ in range(200):
        res = client.get(f"/runs/{rid}")
        if res.status_code != 202:
            break
        time.sleep(0.1)
    assert res.status_code == 200 and res.json()["status"] in ("ok", "partial")
    assert any(x["run_id"] == rid for x in client.get("/runs").json()["runs"])


def test_validation_report_endpoint(sample_run):
    r = client.get(f"/runs/{sample_run['run_id']}/validation?limit=5").json()
    assert r["selected"]["n"] == 1000 and r["selected"]["verdicts"] == {"PASS": 1000}
    assert r["candidates"]["generated"] > 1000 and r["candidates"]["semantic_duplicates"]["inverse_compare"] > 0
    assert len(r["claims"]) == 5 and all({"verdict", "checks", "reasons", "evidence"} <= set(c) for c in r["claims"])
    assert client.get(f"/runs/{sample_run['run_id']}/validation?verdict=FAIL").json()["total"] == 0
    m = client.get(f"/runs/{sample_run['run_id']}/metrics").json()
    assert m["claims"]["validation"]["selected"]["verdicts"] == {"PASS": 1000}
