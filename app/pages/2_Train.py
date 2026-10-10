import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from core.dataset import current_dataset
from core.metrics import pr_fig, roc_fig
from core.training import MODEL_NAMES, UI_PARAMS, train_and_compare

st.set_page_config(page_title="Train", page_icon="🛠️", layout="wide")
st.title("Modules 2 and 3: Preparation and modelling")

ds = current_dataset()
df, target, positive, group = ds["df"], ds["target"], ds["positive"], ds["group"]
st.caption(f"Dataset: **{ds['name']}** ({len(df):,} rows). Target: `{target}`, positive class: `{positive}`.")

# ------------------------------------------------------------------ Module 2
st.subheader("Preparation")
candidates = [c for c in df.columns if c not in (target, group)]
features = st.multiselect("Explanatory variables", candidates, default=candidates,
                          help="Remove variables to test their influence (for example the sensitive ones: race, gender, age).")
c1, c2, c3 = st.columns(3)
test_size = c1.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
use_cw = c2.checkbox("Class weights (imbalance)", value=True,
                     help="Errors on the rare positive class count more during training.")
scale = c3.checkbox("Standardise numeric variables", value=True)
if len(df) > 5000:
    sample = st.slider("Rows used (smaller = faster)", 5000, len(df), min(20000, len(df)), 1000)
else:
    sample = len(df)
st.caption("Missing values: median (numeric) / most frequent (categorical). Categorical variables: one-hot encoding. "
           + ("The split is done **by patient** so that no patient is in both sets." if group else "The split is stratified on the target."))

# ------------------------------------------------------------------ Module 3
st.subheader("Models")
selected = st.multiselect("Models to train", MODEL_NAMES, default=["Logistic Regression", "Random Forest", "XGBoost"])
tune = st.checkbox("Optimise hyperparameters with random search (slower)", value=False)
n_iter = st.slider("Random-search trials per model", 3, 20, 6) if tune else 0

params = {}
if not tune:
    for name in selected:
        with st.expander(f"{name}: hyperparameters", expanded=False):
            p = {}
            for spec in UI_PARAMS[name]:
                if spec["kind"] == "int":
                    p[spec["key"]] = st.slider(spec["label"], spec["min"], spec["max"], spec["default"], spec["step"], key=f"{name}_{spec['key']}")
                else:
                    p[spec["key"]] = st.select_slider(spec["label"], spec["options"], value=spec["default"], key=f"{name}_{spec['key']}")
            params[name] = p
else:
    st.info("With random search, hyperparameters are chosen automatically (scored on PR-AUC with cross-validation).")

if st.button("Train and compare", type="primary", disabled=not selected or not features):
    bar = st.progress(0.0, text="Starting...")
    try:
        out = train_and_compare(df, target, positive, group, features, selected, params, test_size=test_size,
                                use_class_weight=use_cw, sample_size=sample, tune=tune, n_iter=n_iter, scale=scale,
                                progress=lambda f, m: bar.progress(f, text=m))
        st.session_state["trained"] = out
        bar.empty()
    except Exception as e:
        bar.empty()
        st.error(f"Training failed: {e}")

out = st.session_state.get("trained")
if out:
    st.subheader("Comparison table")
    st.caption(f"{out['n_train']:,} training rows, {out['n_test']:,} test rows, positive rate in test: {out['positive_rate_test']:.1%}. "
               "Precision, recall and F1 are computed at threshold 0.5 (change it on the Evaluation page).")
    st.dataframe(out["table"].round(3), width="stretch")
    if tune:
        st.write("Selected hyperparameters:", out["params"])
    a, b = st.columns(2)
    a.plotly_chart(roc_fig(out["y_test"], out["probas"]), width="stretch")
    b.plotly_chart(pr_fig(out["y_test"], out["probas"]), width="stretch")
    st.success("Open the **Evaluation** page to explore the threshold and the confusion matrix of these models.")
