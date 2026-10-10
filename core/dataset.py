"""The dataset currently used by the app (default = prepared hospital data, or an uploaded CSV)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.artifacts import load_all, load_meta


def default_dataset() -> dict:
    meta = load_meta()
    return {"name": "Diabetes 130-US Hospitals (prepared)", "df": load_all(), "target": meta["target"],
            "positive": 1, "group": meta["group_col"]}


def current_dataset() -> dict:
    """Dataset chosen on the Data page, or the default one if the page was not visited yet."""
    return st.session_state.get("dataset") or default_dataset()


@st.cache_data(show_spinner="Downloading the raw dataset from UCI...")
def fetch_uci_raw() -> pd.DataFrame:
    """Automatic download of the raw dataset (UCI id 296). Needs internet access."""
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=296)
    parts = [p for p in (d.data.ids, d.data.features, d.data.targets) if p is not None]
    return pd.concat(parts, axis=1)
