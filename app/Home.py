import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from core.client import api_get, api_url

st.set_page_config(page_title="Hospital readmission: explainable classification", page_icon="🏥", layout="wide")

st.title("🏥 30-day hospital readmission risk")
st.markdown(
    """
Data Mining challenge 1 (ENSI): an interactive application for **explainable binary classification**
on the *Diabetes 130-US Hospitals* data (1999-2008). The goal is to flag diabetic patients at risk of being
**readmitted within 30 days**, and to understand why.

| Page | Module | Content |
|---|---|---|
| **1 Data** | 1 | Dataset, statistics, missing values, class balance, exploratory charts, CSV upload |
| **2 Train** | 2 and 3 | Choice of variables, split, models and hyperparameters, optional random search |
| **3 Evaluation** | 4 | Metrics, ROC / PR curves, confusion matrix, live threshold slider |
| **4 Predict** | 6 | Risk score for one patient, computed by the FastAPI service |
| **5 Explainable AI** | 5 | SHAP global feature importance, beeswarm and patient-level waterfall explanations |

Use the pages in the left sidebar. Start this multipage application with
`streamlit run app/Home.py`; do not run an individual file from `app/pages/`
directly.
"""
)

st.subheader("API status")
st.caption(f"API address: {api_url()}")
try:
    h = api_get("/health")
    st.success(f"API reachable. Models loaded: {', '.join(h['models_loaded'])}")
    for w in h.get("version_warnings", []):
        st.warning(f"Library version difference: {w}")
except (requests.RequestException, KeyError, TypeError) as e:
    st.error("The API is not reachable. Start it with `uvicorn api.main:app --port 8000` (the Predict page needs it).")
    st.caption(str(e)[:200])
