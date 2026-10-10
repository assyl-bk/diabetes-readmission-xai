"""Loading of everything produced by the Kaggle notebooks (models, tables, metadata).

Files are expected in ./artifacts (or in the folder given by the ARTIFACTS_DIR environment variable).
Functions are cached: do NOT modify the returned objects in place (use .copy() first).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))

# These columns are codes (categories), not quantities: always read them as text.
ID_CODE_COLS = ["admission_type_id", "admission_source_id", "discharge_disposition_id"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


@lru_cache
def load_meta() -> dict:
    return json.loads((ARTIFACTS_DIR / "meta.json").read_text())


@lru_cache
def load_results() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "results.csv", index_col=0)


@lru_cache
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")


@lru_cache
def load_split(which: str) -> pd.DataFrame:
    """which = 'train' or 'test' (cleaned tables saved by the preprocessing notebook)."""
    return pd.read_csv(ARTIFACTS_DIR / f"{which}.csv", dtype={c: str for c in ID_CODE_COLS})


def load_all() -> pd.DataFrame:
    return pd.concat([load_split("train"), load_split("test")], ignore_index=True)


def available_models() -> list[str]:
    """Models for which a .joblib file exists, in the order of the results table."""
    folder = ARTIFACTS_DIR / "models"
    return [n for n in load_results().index if (folder / f"{slug(n)}.joblib").exists()]


@lru_cache
def load_model(name: str):
    path = ARTIFACTS_DIR / "models" / f"{slug(name)}.joblib"
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load model artifact '{path}': {exc}") from exc


@lru_cache
def build_schema() -> dict:
    """Describe each input feature (used by the API to validate and by the app to build the form)."""
    meta, train = load_meta(), load_split("train")
    schema = {}
    for col in meta["feature_cols"]:
        s = train[col]
        if col in meta["num_cols"]:
            schema[col] = {
                "type": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "default": float(s.median()),
                "integer": bool((s.dropna() % 1 == 0).all()),
            }
        else:
            counts = s.astype(str).value_counts()
            schema[col] = {
                "type": "categorical",
                "options": list(counts.index),
                "default": str(counts.index[0]),
            }
    return schema
