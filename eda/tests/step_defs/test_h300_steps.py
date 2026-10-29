"""Steps for features/h300.feature. Skips cleanly until eda/h300/run.py / freeze.py have written their files."""
import hashlib
import json
from pathlib import Path

import pytest
from pytest_bdd import given, scenarios, then

ROOT = Path(__file__).resolve().parents[3]
H300_OUT = ROOT / "eda" / "h300" / "out"
MODELS = ROOT / "eda" / "models"

scenarios("h300.feature")


@given("the saved h300 model manifests", target_fixture="manifests")
def manifests():
    found = sorted(MODELS.glob("h300_*.manifest.json")) + sorted(MODELS.glob("practice_cv_best*.manifest.json"))
    if not found:
        pytest.skip("no saved h300 model yet (run eda/h300/run.py)")
    return [json.loads(p.read_text(encoding="utf-8")) for p in found]


@then("every manifest reports all gates passed and a passing permutation test")
def gates_pass(manifests):
    for m in manifests:
        g = m["gates"]
        assert g["all_kept_steps_pass_a_b_c_d"], f"{m['name']}: a kept step failed a gate"
        assert g["permutation_pass"], f"{m['name']}: permutation test failed"


@then("every manifest's robustness check passed")
def robust(manifests):
    for m in manifests:
        assert m["robustness"].get("passed"), f"{m['name']}: {m['robustness']}"


@given("the h300 lockbox log", target_fixture="log")
def lockbox_log():
    p = H300_OUT / "lockbox_log.json"
    if not p.exists():
        pytest.skip("no lockbox look yet")
    return json.loads(p.read_text())


@then("no regime and synthetic version has more than one lockbox look")
def once(log):
    keys = [(e["regime"], e.get("synth")) for e in log]
    dup = {k for k in keys if keys.count(k) > 1}
    assert not dup, f"lockbox looked at more than once for {dup}"


@given("the frozen models", target_fixture="frozen")
def frozen():
    d = {n: MODELS / n for n in ("FINAL", "FALLBACK")}
    if not all((p / "manifest.json").exists() for p in d.values()):
        pytest.skip("models not frozen yet (run eda/h300/freeze.py)")
    return d


@then("FINAL and FALLBACK each have a manifest with a sha256 that matches the model file")
def sha_ok(frozen):
    for n, p in frozen.items():
        m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        assert m["sha256"] == hashlib.sha256((p / "model.joblib").read_bytes()).hexdigest(), n
        assert m["generalisation_checks"], n
