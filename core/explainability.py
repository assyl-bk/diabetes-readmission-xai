
"""SHAP and LIME helpers shared by the API and Streamlit explanation page."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


def positive_probability(model, X):
    """Return the predicted score for class 1."""
    probabilities = model.predict_proba(X)
    return np.asarray(probabilities)[:, 1]


def create_shap_explainer(
    model,
    X_background,
    model_name: str,
):
    """
    Select the SHAP explainer according to
    the model family.

    X_background must be in the same feature
    space expected by model.
    """
    estimator = model.named_steps.get("clf", model)
    preprocessor = model.named_steps.get("pre")
    transformed_background = (
        preprocessor.transform(X_background)
        if preprocessor is not None
        else X_background
    )
    feature_names = (
        list(preprocessor.get_feature_names_out())
        if preprocessor is not None
        else list(X_background.columns)
    )
    transformed_background = pd.DataFrame(
        transformed_background,
        columns=feature_names,
        index=X_background.index,
    )
    name = model_name.lower().replace("_", " ")

    if name in [
        "xgboost",
        "random forest",
        "decision tree"
    ]:
        return shap.TreeExplainer(estimator), transformed_background

    if name in [
        "logistic regression",
        "logistic regression",
    ]:
        return shap.LinearExplainer(
            estimator,
            transformed_background,
        ), transformed_background

    # Model-agnostic explanation for kNN
    # or other supported classifiers.
    def predict_positive(X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(
            X,
            columns=feature_names,
        )
        return positive_probability(estimator, X_df)

    return shap.Explainer(
        predict_positive,
        transformed_background,
        algorithm="permutation"
    ), transformed_background


def calculate_shap(
    model,
    X_background,
    X_explain,
    model_name
):
    explainer, transformed_background = create_shap_explainer(
        model, X_background, model_name
    )
    preprocessor = model.named_steps.get("pre")
    transformed_explain = (
        preprocessor.transform(X_explain)
        if preprocessor is not None
        else X_explain
    )
    transformed_explain = pd.DataFrame(
        transformed_explain,
        columns=transformed_background.columns,
        index=X_explain.index,
    )
    values: Any = explainer(transformed_explain)

    # Some classifiers return one explanation
    # per class. Keep the positive class.
    shap_array = np.asarray(values.values)
    if shap_array.ndim == 3:
        values.values = shap_array[:, :, 1]
    base_array = np.asarray(values.base_values)
    if base_array.ndim == 2:
        values.base_values = base_array[:, 1]
    values.feature_names = list(transformed_explain.columns)

    return values


def calculate_global_importance(shap_values):
    """
    Mean absolute SHAP contribution
    for every feature.
    """
    importance = np.abs(np.asarray(shap_values.values)).mean(axis=0)
    feature_names = shap_values.feature_names or [
        f"feature_{i}" for i in range(len(importance))
    ]

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(
        "importance",
        ascending=False
    )


def create_lime_explainer(
    X_background,
    categorical_features=None
):
    return LimeTabularExplainer(
        training_data=X_background.to_numpy(),
        feature_names=list(X_background.columns),
        class_names=[
            "No readmission within 30 days",
            "Readmission within 30 days"
        ],
        mode="classification",
        categorical_features=categorical_features,
        discretize_continuous=True,
        random_state=42
    )


def explain_with_lime(
    model,
    X_background,
    patient,
    categorical_features=None,
    num_features=10
):
    lime_explainer = create_lime_explainer(
        X_background,
        categorical_features
    )

    def predict_fn(X):
        X_df = pd.DataFrame(
            X,
            columns=X_background.columns
        )
        return model.predict_proba(X_df)

    explanation = lime_explainer.explain_instance(
        data_row=patient.to_numpy(),
        predict_fn=predict_fn,
        num_features=num_features,
        labels=(1,),
        num_samples=1000
    )

    return explanation


def compute_shap(model, X_background, X_explain):
    """Compatibility wrapper used by the Streamlit explanation page."""
    return calculate_shap(model, X_background, X_explain, "XGBoost")


def plot_global_importance(shap_values):
    import matplotlib.pyplot as plt

    importance = calculate_global_importance(shap_values).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    return fig


def plot_beeswarm(shap_values):
    import matplotlib.pyplot as plt

    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    return plt.gcf()


def plot_patient_waterfall(shap_values, patient_index: int):
    import matplotlib.pyplot as plt

    index = int(patient_index)
    base_values = np.asarray(shap_values.base_values)
    patient_values = shap.Explanation(
        values=np.asarray(shap_values.values)[index],
        base_values=float(base_values[index]),
        data=None if shap_values.data is None else np.asarray(shap_values.data)[index],
        feature_names=shap_values.feature_names,
    )
    shap.plots.waterfall(patient_values, max_display=15, show=False)
    return plt.gcf()
