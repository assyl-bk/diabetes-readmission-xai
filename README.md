# Hospital readmission risk: explainable classification (Data Mining challenge 1, ENSI)

Binary classification of **30-day readmission** on *Diabetes 130-US Hospitals* (UCI id 296).

- `notebooks/`: preprocessing (Modules 1-2) and modelling (Modules 3-4), run on Kaggle
- `core/`: shared code (artifact loading, metrics, interactive training)
- `api/main.py`: FastAPI service (Module 6)
- `app/`: Streamlit application (Home + 4 pages)
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

## API endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, loaded models, library version check |
| GET | `/models` | models and their test metrics |
| GET | `/schema` | description of the input features |
| POST | `/predict` | risk score and decision for one or more patients |
| GET | `/explain/global` | built-in feature importance (SHAP / LIME endpoints come with Module 5) |

## Deployment
- **API** (Render or Hugging Face Spaces): start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, install `requirements-api.txt`.
- **App** (Streamlit Community Cloud): main file `app/Home.py`, requirements `requirements.txt`, and in *Secrets* add `API_URL = "https://your-api-url"`.

## Links and screenshots
(to fill in)
