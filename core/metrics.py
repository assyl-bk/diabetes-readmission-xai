"""Metrics and figures for Module 4 (evaluation), shared by the app pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def metrics_at(y_true, p, thr: float = 0.5) -> dict:
    y_true, p = np.asarray(y_true), np.asarray(p)
    pred = (p >= thr).astype(int)
    prevalence = y_true.mean()
    pr_auc = average_precision_score(y_true, p)
    return {
        "AUC-ROC": roc_auc_score(y_true, p),
        "PR-AUC": pr_auc,
        "PR-AUC lift": pr_auc / prevalence if prevalence > 0 else np.nan,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Flagged %": pred.mean() * 100,
    }


def comparison_table(y_true, probas: dict, thr: float = 0.5) -> pd.DataFrame:
    table = pd.DataFrame({name: metrics_at(y_true, p, thr) for name, p in probas.items()}).T
    return table.sort_values("PR-AUC", ascending=False)


def roc_fig(y_true, probas: dict) -> go.Figure:
    fig = go.Figure()
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {roc_auc_score(y_true, p):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="random"))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=480)
    return fig


def pr_fig(y_true, probas: dict) -> go.Figure:
    fig = go.Figure()
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (PR-AUC {average_precision_score(y_true, p):.3f})"))
    base = float(np.mean(y_true))
    fig.add_hline(y=base, line_dash="dash", line_color="grey", annotation_text=f"baseline {base:.3f}")
    fig.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision", height=480)
    return fig


def confusion_fig(y_true, p, thr: float, name: str = "") -> go.Figure:
    pred = (np.asarray(p) >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["predicted: no", "predicted: yes"], y=["actual: no", "actual: yes"],
                    title=f"Confusion matrix: {name} (threshold {thr:.2f})")
    fig.update_layout(height=420, coloraxis_showscale=False)
    return fig


def threshold_fig(y_true, p) -> go.Figure:
    rows = []
    for t in np.round(np.arange(0.02, 0.99, 0.01), 2):
        pred = (np.asarray(p) >= t).astype(int)
        rows.append({"threshold": t,
                     "precision": precision_score(y_true, pred, zero_division=0),
                     "recall": recall_score(y_true, pred, zero_division=0),
                     "f1": f1_score(y_true, pred, zero_division=0)})
    d = pd.DataFrame(rows).melt("threshold", var_name="metric", value_name="value")
    return px.line(d, x="threshold", y="value", color="metric", title="Precision / recall / F1 vs threshold")
