# Hospital readmission risk: explainable classification (Data Mining challenge 1, ENSI)

Binary classification of **30-day readmission** on *Diabetes 130-US Hospitals* (UCI id 296).

- `notebooks/`: preprocessing (Modules 1-2) and modelling (Modules 3-4), run on Kaggle
- `core/`: shared code (artifact loading, metrics, interactive training)
- `core/explainability.py`: shared SHAP and LIME explainability utilities
- `api/main.py`: FastAPI service (Module 6)
- `api/xai_routes.py`: FastAPI routes for global and patient-level SHAP explanations
- `app/`: Streamlit application (Home + 5 pages, including Explainable AI)
- `artifacts/`: files produced on Kaggle (models, tables, metadata)

## Run locally
```bash
# 1. environment (once)
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. put the Kaggle outputs in artifacts/ (see artifacts/README.md)

# 3. terminal 1: the API  ->  http://localhost:8000/docs
uvicorn api.main:app --reload --port 8000

# 4. terminal 2: the app  ->  http://localhost:8501
streamlit run app/Home.py
```

The Explainable AI page is available from the left sidebar at the same
Streamlit address. Do not run `app/pages/5_Explainable_AI.py` directly, because
that starts a separate Streamlit application.

## Explainable AI

The Explainable AI module helps interpret the readmission-risk models instead
of presenting their predictions as black boxes. It is available as
**5 Explainable AI** in the sidebar of the Streamlit application:

```text
streamlit run app/Home.py
```

The page uses the trained pipelines and the prepared train/test artifacts
already stored in `artifacts/`. It excludes the target and patient identifier
from the explanation inputs, samples up to 100 training rows as a background
dataset, and explains up to 100 test patients at a time.

### Global explanations

The **Global explanations** tab contains:

- **Mean absolute SHAP feature importance**: ranks features by the average
  absolute contribution across the selected test sample. Larger values mean
  that the feature has a stronger influence on the model output for that
  sample; they do not indicate whether the influence is positive or negative.
- **SHAP beeswarm plot**: shows the distribution of feature contributions for
  individual observations. The horizontal position represents the contribution
  to the model output, while the feature ordering reflects overall importance.

Categorical variables are displayed using the names created by the fitted
one-hot encoder, for example `cat__gender_Female` or
`cat__discharge_disposition_id_1`. Numeric variables retain their transformed
feature names, such as `num__number_inpatient`.

### Individual explanations

The **Individual explanation** tab lets the user select a patient from the
test sample and displays a SHAP waterfall plot. The waterfall starts at the
model's base value and shows the most influential feature contributions for
that patient. Contributions moving the output upward increase the model score;
contributions moving it downward reduce the score.

The model score should be interpreted as a risk-ranking score rather than a
calibrated probability. This is especially important here because the models
were trained with class weighting to account for the relatively small
positive class.

### Explainer selection

`core/explainability.py` selects an explainer according to the model family:

| Model family | Explainer |
|---|---|
| XGBoost | `shap.TreeExplainer` |
| Random Forest | `shap.TreeExplainer` |
| Decision Tree | `shap.TreeExplainer` |
| Logistic Regression | `shap.LinearExplainer` |
| kNN | model-agnostic SHAP permutation explainer |

For pipeline models, the fitted preprocessing step is applied before SHAP is
calculated. The resulting explanations therefore use the transformed feature
space, including one-hot encoded categorical features. For tree models that
return one set of values per class, the implementation keeps the positive
readmission class. Patient-level waterfall plots are rebuilt with a scalar
base value so they work consistently across all supported classifiers.

The module also contains LIME helpers (`create_lime_explainer` and
`explain_with_lime`) for model-agnostic local explanations. They are available
for future UI/API expansion but are not required to display the current
Streamlit page.

## API endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, loaded models, library version check |
| GET | `/models` | models and their test metrics |
| GET | `/schema` | description of the input features |
| POST | `/predict` | risk score and decision for one or more patients |
| GET | `/explain/global` | model-native feature importance for the selected model |
| GET | `/explain/shap/global` | SHAP global importance for a test sample |
| POST | `/explain/shap/local` | SHAP contributions for one test patient |

Example SHAP requests:

```text
GET http://localhost:8000/explain/shap/global?model_name=XGBoost&sample_size=100
```

```json
POST http://localhost:8000/explain/shap/local
{
  "model_name": "XGBoost",
  "patient_index": 0
}
```

The global endpoint accepts a sample size from 1 to 300. The local endpoint
returns the selected model, patient index, risk score, scalar base value, and
the transformed-feature contributions. Both endpoints require the FastAPI
service to be running in the API terminal.

## Deployment
- **API** (Render or Hugging Face Spaces): start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, install `requirements-api.txt`.
- **App** (Streamlit Community Cloud): main file `app/Home.py`, requirements `requirements.txt`, and in *Secrets* add `API_URL = "https://your-api-url"`.
