import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.dataset import default_dataset, fetch_uci_raw

st.set_page_config(page_title="Data", page_icon="📊", layout="wide")
st.title("Module 1: Data")


def explore(df: pd.DataFrame, target: str, positive, key: str):
    """Preview, descriptive statistics, missing values, class balance, exploratory charts."""
    y = (df[target].astype(str) == str(positive)).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Positive class", f"{y.mean():.1%}")

    t1, t2, t3, t4, t5 = st.tabs(["Preview", "Statistics", "Missing values", "Class balance", "Exploration"])
    with t1:
        st.dataframe(df.head(100), width="stretch")
    with t2:
        st.dataframe(df.describe(include="all").T, width="stretch")
    with t3:
        miss = (df.isna().mean() * 100).round(2).rename("% missing").to_frame()
        miss = miss[miss["% missing"] > 0].sort_values("% missing", ascending=False)
        if miss.empty:
            st.success("No missing values.")
        else:
            st.dataframe(miss, width="stretch")
    with t4:
        bal = y.map({0: "negative", 1: "positive"}).value_counts()
        st.plotly_chart(px.pie(values=bal.values, names=bal.index, title="Class balance"), width="stretch")
        if y.mean() < 0.3 or y.mean() > 0.7:
            st.info("The classes are imbalanced: accuracy is misleading, so the app reports PR-AUC, recall and F1 on the positive class.")
    with t5:
        others = [c for c in df.columns if c != target]
        col = st.selectbox("Variable", others, key=f"var_{key}")
        s = df[col]
        left, right = st.columns(2)
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
            grouped = pd.qcut(s, q=min(8, s.nunique()), duplicates="drop").astype(str)
            left.plotly_chart(px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}"), width="stretch")
        else:
            grouped = s.astype(str)
            left.plotly_chart(px.histogram(df.assign(_v=grouped).sort_values("_v"), x="_v", title=f"Distribution of {col}")
                              .update_xaxes(title=col), width="stretch")
        rate = pd.DataFrame({"g": grouped, "y": y}).groupby("g")["y"].agg(rate="mean", n="size").reset_index()
        fig = px.bar(rate, x="g", y="rate", hover_data=["n"], title=f"Positive-class rate by {col}")
        fig.add_hline(y=y.mean(), line_dash="dash", annotation_text=f"overall {y.mean():.1%}")
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title=col)
        right.plotly_chart(fig, width="stretch")

        num = df.select_dtypes(include="number").drop(columns=[c for c in ["patient_nbr"] if c in df.columns], errors="ignore")
        if num.shape[1] >= 2:
            st.plotly_chart(px.imshow(num.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                      title="Correlation matrix (numeric variables)", aspect="auto"), width="stretch")


source = st.radio("Data source", ["Prepared hospital data (used for training)", "Raw UCI download (automatic)", "Upload my own CSV"],
                  horizontal=True)

if source.startswith("Prepared"):
    ds = default_dataset()
    st.session_state["dataset"] = ds
    st.caption("Cleaned table produced by the preprocessing notebook (train + test). Target: `target` = readmitted within 30 days.")
    explore(ds["df"], ds["target"], ds["positive"], "prepared")

elif source.startswith("Raw"):
    st.caption("Automatic download with `ucimlrepo` (no manual step). This is the raw table, before cleaning.")
    try:
        raw = fetch_uci_raw()
        raw = raw.replace("?", np.nan)
        st.info("Raw target `readmitted` has 3 values (NO, >30, <30). The project reduces it to 2 classes: `<30` = positive.")
        explore(raw, "readmitted", "<30", "raw")
    except Exception as e:
        st.error("Could not download the dataset (internet access or UCI server problem). "
                 "Use the prepared data or upload a CSV.")
        st.caption(str(e)[:200])

else:
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        up = pd.read_csv(file)
        target = st.selectbox("Target column", up.columns)
        values = up[target].dropna().unique().tolist()
        if len(values) > 30:
            st.warning("This column has many distinct values: choose a categorical or 0/1 target.")
        else:
            positive = st.selectbox("Value of the positive class", values)
            st.session_state["dataset"] = {"name": file.name, "df": up, "target": target, "positive": positive, "group": None}
            st.success(f"Dataset '{file.name}' will be used on the Train page (positive class = {positive}).")
            explore(up, target, positive, "upload")
    else:
        st.info("Upload a CSV and choose the target column. The same app then trains models on your data.")
