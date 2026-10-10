import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from core.artifacts import load_meta, load_predictions
from core.metrics import comparison_table, confusion_fig, metrics_at, pr_fig, roc_fig, threshold_fig

st.set_page_config(page_title="Evaluation", page_icon="📈", layout="wide")
st.title("Module 4: Evaluation")

# ---------------------------------------------------------------- choose which models to evaluate
options = ["Models trained on Kaggle (final models of the notebook)"]
if "trained" in st.session_state:
    options.append("Models trained in this app (Train page)")
source = st.radio("Models to evaluate", options, horizontal=True)

if source.startswith("Models trained on Kaggle"):
    pred = load_predictions()
    y = pred["y_true"].to_numpy()
    probas = {c: pred[c].to_numpy() for c in pred.columns if c not in ("patient_nbr", "y_true")}
    default_thr = load_meta().get("operating_threshold", 0.5)
else:
    out = st.session_state["trained"]
    y, probas, default_thr = out["y_test"], out["probas"], 0.5

st.caption(f"{len(y):,} test patients, positive rate {y.mean():.1%} (this is also the PR-AUC baseline of a random model).")

# ---------------------------------------------------------------- threshold slider
st.subheader("Decision threshold")
thr = st.slider("Threshold: patients with a score at or above it are flagged", 0.01, 0.99,
                float(round(min(max(default_thr, 0.01), 0.99), 2)), 0.01)
st.caption("Lowering the threshold catches more true readmissions (higher recall) but flags more patients (lower precision). "
           "The Kaggle default is the threshold minimising a cost where a missed readmission is 5 times worse than a false alarm.")

st.dataframe(comparison_table(y, probas, thr).round(3), width="stretch")

# ---------------------------------------------------------------- curves
a, b = st.columns(2)
a.plotly_chart(roc_fig(y, probas), width="stretch")
b.plotly_chart(pr_fig(y, probas), width="stretch")

# ---------------------------------------------------------------- one model in detail
st.subheader("One model in detail")
name = st.selectbox("Model", list(probas), index=0)
m = metrics_at(y, probas[name], thr)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Recall", f"{m['Recall']:.2f}")
k2.metric("Precision", f"{m['Precision']:.2f}")
k3.metric("F1", f"{m['F1']:.2f}")
k4.metric("Patients flagged", f"{m['Flagged %']:.1f}%")

c, d = st.columns(2)
c.plotly_chart(confusion_fig(y, probas[name], thr, name), width="stretch")
d.plotly_chart(threshold_fig(y, probas[name]), width="stretch")
