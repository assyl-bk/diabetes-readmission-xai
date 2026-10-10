import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from core.client import api_get, api_post

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("Module 6: Predict with the API")
st.caption("This page sends the patient to the FastAPI service, which returns the risk score. Nothing is computed in the page itself.")

try:
    info = api_get("/models")
    schema = api_get("/schema")
except Exception as e:
    st.error("The API is not reachable. Start it in another terminal: `uvicorn api.main:app --port 8000`")
    st.caption(str(e)[:200])
    st.stop()

names = [m["name"] for m in info["models"]]
c1, c2 = st.columns(2)
model = c1.selectbox("Model", names, index=names.index(info["best_model"]) if info["best_model"] in names else 0)
thr = c2.slider("Decision threshold", 0.01, 0.99, float(round(info["operating_threshold"], 2)), 0.01)

# ---------------------------------------------------------------- optional: start from a real test patient
st.subheader("Patient")
start = st.radio("Start from", ["A typical patient", "A real patient of the test set"], horizontal=True)
defaults = {k: v["default"] for k, v in schema.items()}
if start.startswith("A real"):
    from core.artifacts import load_split
    test = load_split("test")
    row = st.number_input("Row number in the test set", 0, len(test) - 1, 0)
    r = test.iloc[int(row)]
    defaults = {k: r[k] for k in schema}
    st.caption(f"Actual outcome of this patient: {'readmitted within 30 days' if r['target'] == 1 else 'not readmitted within 30 days'}")

# ---------------------------------------------------------------- form built from the API schema
record = {}
cols = st.columns(3)
for i, (feat, spec) in enumerate(schema.items()):
    with cols[i % 3]:
        if spec["type"] == "numeric":
            if spec["integer"]:
                record[feat] = st.number_input(feat, int(spec["min"]), int(spec["max"]), int(defaults[feat]), step=1, key=f"in_{feat}_{start}_{defaults[feat]}")
            else:
                record[feat] = st.number_input(feat, spec["min"], spec["max"], float(defaults[feat]), key=f"in_{feat}_{start}_{defaults[feat]}")
        else:
            opts = spec["options"]
            idx = opts.index(str(defaults[feat])) if str(defaults[feat]) in opts else 0
            record[feat] = st.selectbox(feat, opts, index=idx, key=f"in_{feat}_{start}_{defaults[feat]}")

if st.button("Compute risk score", type="primary"):
    try:
        res = api_post("/predict", {"model": model, "threshold": thr, "records": [record]})
        p = res["predictions"][0]
        st.subheader("Result")
        a, b = st.columns(2)
        a.metric("Risk score", f"{p['probability']:.1%}")
        b.metric("Decision", p["label"])
        st.progress(min(max(p["probability"], 0.0), 1.0))
        st.caption(f"Flagged when the score is at least {res['threshold']:.2f}. The score comes from a model trained with class weights: "
                   "treat it as a risk ranking, not as the exact probability of readmission.")
        for w in res["warnings"]:
            st.warning(w)
        with st.expander("What the API returned"):
            st.json(res)
    except Exception as e:
        st.error(f"API error: {e}")

# ---------------------------------------------------------------- global importance from the API
st.subheader("What the model relies on (built-in importance)")
try:
    g = api_get("/explain/global", model=model, top=12)
    d = pd.DataFrame(g["features"])
    st.plotly_chart(px.bar(d.iloc[::-1], x="value", y="feature", orientation="h", title=f"{model}: {g['kind']}"), width="stretch")
except Exception:
    st.info(f"{model} has no built-in importance. SHAP / permutation importance (Module 5) will cover every model.")
