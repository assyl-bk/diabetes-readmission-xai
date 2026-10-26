import matplotlib.pyplot as plt
import streamlit as st

from core.artifacts import available_models, load_model, load_split
from core.explainability import (
    calculate_shap,
    plot_beeswarm,
    plot_global_importance,
    plot_patient_waterfall,
)

st.set_page_config(page_title="Explainable AI", page_icon="🔍", layout="wide")
st.title("Explainable AI - Diabetes Readmission")

models = available_models()
if not models:
    st.error("No trained model artifacts are available.")
    st.stop()

model_name = st.selectbox("Model", models)
train = load_split("train")
test = load_split("test")
features = [c for c in train.columns if c not in {"target", "patient_nbr"}]
background = train[features].sample(min(100, len(train)), random_state=42)
explain = test[features].head(100)

try:
    shap_values = calculate_shap(
        load_model(model_name),
        background,
        explain,
        model_name,
    )
except (ImportError, RuntimeError, TypeError, ValueError) as exc:
    st.error(f"Could not calculate SHAP explanations: {exc}")
    st.stop()

tab1, tab2 = st.tabs(["Global explanations", "Individual explanation"])

with tab1:
    st.subheader("Feature importance")
    fig = plot_global_importance(shap_values)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("SHAP beeswarm")
    fig = plot_beeswarm(shap_values)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    patient_index = st.selectbox("Select a patient", range(len(explain)))
    fig = plot_patient_waterfall(shap_values, patient_index)
    st.pyplot(fig)
    plt.close(fig)
