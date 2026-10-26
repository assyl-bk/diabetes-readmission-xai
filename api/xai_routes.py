from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.artifacts import available_models, load_model, load_split
from core.explainability import (
    calculate_global_importance,
    calculate_shap,
)

router = APIRouter(
    prefix="/explain",
    tags=["Explainable AI"]
)


def get_xai_resources(model_name: str):
    if model_name not in available_models():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Available: {available_models()}",
        )

    model = load_model(model_name)
    background = load_split("train").drop(columns=["target", "patient_nbr"])
    test = load_split("test").drop(columns=["target", "patient_nbr"])
    return model, background, test


class LocalExplanationRequest(BaseModel):
    model_name: str
    patient_index: int


@router.get("/shap/global")
def global_shap(
    model_name: str = "XGBoost",
    sample_size: int = 100
):
    if sample_size < 1 or sample_size > 300:
        raise HTTPException(
            status_code=400,
            detail="sample_size must be between 1 and 300"
        )

    model, background, test = get_xai_resources(model_name)

    X_sample = test.iloc[:sample_size]

    shap_values = calculate_shap(
        model,
        background,
        X_sample,
        model_name
    )

    importance = calculate_global_importance(
        shap_values
    )

    return {
        "model": model_name,
        "sample_size": len(X_sample),
        "features": importance.to_dict(
            orient="records"
        )
    }


@router.post("/shap/local")
def local_shap(
    request: LocalExplanationRequest
):
    model, background, test = get_xai_resources(request.model_name)

    index = request.patient_index

    if index < 0 or index >= len(test):
        raise HTTPException(
            status_code=404,
            detail="Patient index not found"
        )

    patient = test.iloc[[index]]

    shap_values = calculate_shap(
        model,
        background,
        patient,
        request.model_name
    )

    return {
        "model": request.model_name,
        "patient_index": index,
        "score": float(
            model.predict_proba(patient)[0, 1]
        ),
        "base_value": float(shap_values.base_values[0]),
        "contributions": [
            {
                "feature": feature,
                "value": float(value)
            }
            for feature, value in zip(shap_values.feature_names, shap_values.values[0])
        ]
    }
