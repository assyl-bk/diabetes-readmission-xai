"""FastAPI service of the project (Module 6).

Run locally:   uvicorn api.main:app --reload --port 8000
Interactive documentation (Swagger): http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core.artifacts import available_models, build_schema, load_meta, load_model, load_results


@asynccontextmanager
async def lifespan(app: FastAPI):
    meta = load_meta()
    app.state.meta = meta
    app.state.schema = build_schema()
    app.state.models = {}
    for name in available_models():
        try:
            app.state.models[name] = load_model(name)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Startup failed while loading '{name}'. "
                "Replace the corresponding file in artifacts/models with a fresh copy "
                "from modeling_outputs.zip."
            ) from exc
    expected = meta.get("versions", {})
    installed = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__}
    app.state.version_warnings = [f"{k}: models trained with {v}, installed {installed[k]}"
                                  for k, v in expected.items() if k in installed and installed[k] != v]
    yield


app = FastAPI(
    title="Hospital readmission risk API",
    description=("Prediction service for the 30-day readmission classifier (Diabetes 130-US Hospitals). "
                 "Data Mining challenge 1, ENSI. Returns a **risk score**, not a calibrated probability: "
                 "the models were trained with class weights because only about 11% of stays are positive."),
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    model: str = Field("XGBoost", description="Model name, see GET /models")
    threshold: float | None = Field(None, ge=0, le=1, description="Decision threshold (default: 0.5)")
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000,
                                          description="One dictionary per patient. Missing features get a default value.")


class Prediction(BaseModel):
    probability: float = Field(..., description="Model score between 0 and 1")
    decision: int = Field(..., description="1 = flagged as high readmission risk at this threshold")
    label: str


class PredictResponse(BaseModel):
    model: str
    threshold: float
    predictions: list[Prediction]
    warnings: list[str]


# ----------------------------------------------------------------------------- helpers
def _get_model(name: str):
    models = app.state.models
    if name not in models:
        raise HTTPException(404, f"Unknown model '{name}'. Available: {list(models)}")
    return models[name]


def _to_frame(records: list[dict], warnings: list[str]) -> pd.DataFrame:
    schema = app.state.schema
    df = pd.DataFrame(records)
    unknown_cols = [c for c in df.columns if c not in schema]
    if unknown_cols:
        warnings.append(f"Ignored unknown fields: {unknown_cols}")
    missing = [c for c in schema if c not in df.columns]
    if missing:
        warnings.append(f"Default values used for: {missing}")
    for col, spec in schema.items():
        if col not in df.columns:
            df[col] = spec["default"]
        df[col] = df[col].where(df[col].notna(), spec["default"])
        if spec["type"] == "numeric":
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                raise HTTPException(422, f"Field '{col}' must be numeric")
        else:
            df[col] = df[col].astype(str)
            unseen = sorted(set(df[col]) - set(spec["options"]))
            if unseen:
                warnings.append(f"Unseen category in '{col}': {unseen} (treated as 'none of the known categories')")
    return df[list(schema)]


# ----------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["service"])
def health():
    """Service status and library version check."""
    return {"status": "ok", "models_loaded": list(app.state.models), "version_warnings": app.state.version_warnings}


@app.get("/models", tags=["models"])
def list_models():
    """Available models with their test-set metrics (computed at threshold 0.5 in the notebook)."""
    results = load_results()
    meta = app.state.meta
    return {
        "best_model": meta["best_model"],
        "default_threshold": meta["default_threshold"],
        "operating_threshold": meta["operating_threshold"],
        "positive_rate_test": meta["positive_rate_test"],
        "models": [{"name": n, "metrics": {k: (None if pd.isna(v) else float(v)) for k, v in results.loc[n].items()}}
                   for n in app.state.models],
    }


@app.get("/schema", tags=["models"])
def feature_schema():
    """Description of every input feature (type, range or allowed values, default)."""
    return app.state.schema


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    """Risk score and decision for one or several patients."""
    pipe = _get_model(req.model)
    warnings: list[str] = []
    X = _to_frame(req.records, warnings)
    thr = app.state.meta["default_threshold"] if req.threshold is None else req.threshold
    proba = pipe.predict_proba(X)[:, 1]
    preds = [Prediction(probability=float(p), decision=int(p >= thr),
                        label="high readmission risk" if p >= thr else "lower readmission risk") for p in proba]
    return PredictResponse(model=req.model, threshold=thr, predictions=preds, warnings=warnings)


@app.get("/explain/global", tags=["explanation"])
def global_importance(model: str = Query("XGBoost"), top: int = Query(15, ge=1, le=100)):
    """Model-native global importance: coefficients (logistic regression) or impurity/gain importance (trees)."""
    pipe = _get_model(model)
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    if hasattr(clf, "coef_"):
        values, kind = np.ravel(clf.coef_), "coefficient on standardised / one-hot inputs (positive = raises the risk score)"
    elif hasattr(clf, "feature_importances_"):
        values, kind = np.asarray(clf.feature_importances_), "impurity / gain importance (always positive, no direction)"
    else:
        raise HTTPException(400, f"{model} has no built-in importance: use permutation importance or SHAP (Module 5).")
    order = np.argsort(-np.abs(values))[:top]
    return {"model": model, "kind": kind, "features": [{"feature": str(names[i]), "value": float(values[i])} for i in order]}
