"""Small helper used by the Streamlit app to call the FastAPI service."""
from __future__ import annotations

import os

import requests
import streamlit as st


def api_url() -> str:
    url = os.getenv("API_URL")
    if not url:
        try:
            url = st.secrets.get("API_URL")        # Streamlit Cloud: set it in the app's Secrets
        except Exception:
            url = None
    return (url or "http://127.0.0.1:8000").rstrip("/")


def api_get(path: str, **params):
    r = requests.get(f"{api_url()}{path}", params=params, timeout=90)   # long timeout: free hosts wake up slowly
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{api_url()}{path}", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()
