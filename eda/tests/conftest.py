"""Shared fixtures for the research test suite (eda/tests).

Everything here reads the committed research outputs in eda/out and eda/splits.csv; nothing is written outside
pytest's tmp_path. Expensive tables are loaded once per session.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
EDA = ROOT / "eda"
OUT = EDA / "out"
for p in (str(ROOT), str(EDA)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module(name):
    """Import an eda script whose file name starts with a digit (e.g. '02_rules')."""
    return importlib.import_module(name)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def rows():
    return pd.read_pickle(OUT / "rows.pkl")


@pytest.fixture(scope="session")
def facts():
    return pd.read_pickle(OUT / "facts.pkl")


@pytest.fixture(scope="session")
def features():
    return pd.read_pickle(OUT / "features.pkl")


@pytest.fixture(scope="session")
def splits():
    return pd.read_csv(EDA / "splits.csv")


@pytest.fixture(scope="session")
def improve_labels():
    return pd.read_csv(OUT / "improve_labels.csv")


@pytest.fixture(scope="session")
def improve_neuro():
    p = OUT / "improve_neuro.csv"
    if not p.exists():
        pytest.skip("eda/out/improve_neuro.csv not produced yet")
    return pd.read_csv(p)


def facts_frame(rows):
    """Small synthetic facts frame in the shape eda/improve/timeline.Timeline expects.

    rows: list of (id, page, date, value) for one (subject, attribute) pair.
    """
    df = pd.DataFrame(rows, columns=["id", "page", "date", "vkey"])
    df["date"] = pd.to_datetime(df["date"])
    df["subject"] = "Hooly Test"
    df["attribute"] = "price"
    return df


@pytest.fixture
def make_facts():
    return facts_frame


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
