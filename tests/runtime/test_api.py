import io

from fastapi.testclient import TestClient

from app.api.main import STAGE_ORDER, app

client = TestClient(app)


def test_architecture_shape():
    a = client.get("/architecture").json()
    ids = {n["id"] for n in a["nodes"]}
    assert a["nodes"] and all({"source", "target"} <= set(e) for e in a["edges"])
    assert all(e["source"] in ids and e["target"] in ids for e in a["edges"])


def test_run_sample_shape():
    r = client.post("/run", json={"use": "sample"}).json()
    assert r["stages"] == STAGE_ORDER
    assert set(r["timings"]) == set(STAGE_ORDER)
    assert len(r["labels"]) == len(r["rows"]) > 0
    assert r["claims_total"] == 1000 and len(r["claims"]) == 50
    assert all(lb["label"] in ("true", "false", "unsure") for lb in r["labels"])


def test_run_upload_semicolon_csv():
    data = "id;text;date;label\nx-1;Hooly AI Pro costs €59 per month.;2026-10-01;\n".encode("utf-8-sig")
    r = client.post("/run", files={"file": ("s.csv", io.BytesIO(data), "text/csv")}).json()
    assert [lb["id"] for lb in r["labels"]] == ["x-1"]


def test_metrics_has_runtime():
    client.post("/run", json={"use": "sample"})
    assert "runtime" in client.get("/metrics").json()
