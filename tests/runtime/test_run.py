import csv
import os
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def universe(tmp_path):
    u, o = tmp_path / "universe", tmp_path / "output"
    u.mkdir()
    o.mkdir()
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "input.py"),
                    os.path.join(ROOT, "data", "practice.csv"), str(u / "universe.csv")], check=True)
    shutil.copy(os.path.join(ROOT, "data", "lexicon.json"), u)
    return u, o


def _check(u, o):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_output.py"), str(o),
                           os.path.join(ROOT, "data", "practice.csv"), str(u / "universe.csv")],
                          capture_output=True, text=True)


def test_main_passes_upstream_checker(universe):
    from app.run import main

    u, o = universe
    report = main(str(u), str(o), time.time() + 600)
    r = _check(u, o)
    assert r.returncode == 0, r.stdout
    assert report["total_s"] < 120
    assert not [f for f in os.listdir(o) if f != "output.csv"], "temp files left behind"


def test_survives_crashing_model(universe, monkeypatch, tmp_path):
    """A P3 module that raises must not cost the table."""
    u, o = universe
    pkg = tmp_path / "fake" / "app"
    for sub in ("extract", "classify"):
        (pkg / sub).mkdir(parents=True)
        (pkg / sub / "__init__.py").write_text(
            f"def {sub}(*a, **k):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    import app

    monkeypatch.setattr(app, "__path__", list(app.__path__) + [str(pkg)])
    for m in ("app.extract", "app.classify"):
        sys.modules.pop(m, None)
    from app.run import main

    main(str(u), str(o), time.time() + 600)
    assert _check(u, o).returncode == 0
    for m in ("app.extract", "app.classify"):
        sys.modules.pop(m, None)


def test_semicolon_and_bom_input(universe):
    from app.run import main

    u, o = universe
    path = u / "universe.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)
    main(str(u), str(o), time.time() + 600)
    assert _check(u, o).returncode == 0


def test_sigterm_leaves_a_table(universe):
    u, o = universe
    env = dict(os.environ, UNIVERSE_DIR=str(u), OUTPUT_DIR=str(o), ARENA_DEADLINE=str(int(time.time()) + 600),
               PYTHONPATH=ROOT)
    p = subprocess.Popen([sys.executable, "-m", "app.run"], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if (o / "output.csv").exists() or p.poll() is not None:
            break
        time.sleep(0.1)
    p.terminate()
    p.wait(timeout=30)
    assert (o / "output.csv").exists()
