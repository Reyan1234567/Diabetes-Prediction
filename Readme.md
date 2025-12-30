# Diabeto API

FastAPI-based microservice that wraps two pre-trained scikit-learn classifiers for diabetes-related risk scoring. Each endpoint accepts rich demographic, lifestyle, and clinical metrics and returns the model's binary prediction.

> **Heads up:** the current responses label the outcome as `"Smoker"` / `"Non-Smoker"` even though the underlying task is diabetes risk. Update the response strings if you want them to reflect the medical outcome instead of smoking status.

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `api.py` | FastAPI application exposing `/diabeto/logistic` (logistic regression) and `/diabeto/tree` (decision tree) endpoints. |
| `clipper.py` | Custom `iqr_clipper` transformer used during model training to cap outliers. |
| `diabeto_LR.joblib` | Serialized logistic regression pipeline. |
| `diabeto_DC.joblib` | Serialized decision-tree pipeline (≈27 MB). |
| `train.csv`, `test.csv`, `result.csv`, `sample_submission.csv` | Tabular datasets used for experimentation (kept for reproducibility). |
| `Diabeto.ipynb` | Notebook used for exploration / training (ignored when running the API). |

## Requirements

- Python 3.10+ (newer versions are fine as long as FastAPI + scikit-learn install cleanly).
- Recommended packages (install via `pip install -r requirements.txt` once you add one, or run the one-liner below):

```bash
pip install fastapi uvicorn[standard] pandas numpy scikit-learn joblib pydantic
```

## Quick start

1. **Clone & set up a virtual env**
   ```powershell
   git clone <repo-url>
   cd playground-series-s5e12
   python -m venv .venv
   .venv\Scripts\activate
   pip install fastapi uvicorn[standard] pandas numpy scikit-learn joblib pydantic
   ```

2. **Serve the API**
   ```powershell
   uvicorn api:app --reload
   ```

   Run this command from the project root (`D:\playground-series-s5e12`). That ensures the relative paths to `diabeto_*.joblib` resolve correctly.

3. **Try it**
   ```bash
   curl -X POST http://127.0.0.1:8000/diabeto/logistic \
        -H "Content-Type: application/json" \
        -d @sample-payload.json
   ```

## Request schema

All endpoints accept the same payload. Enums must match the exact strings shown.

```json
{
  "age": 45,
  "alcohol_consumption_per_week": 2,
  "physical_activity_minutes_per_week": 180,
  "diet_score": 7.5,
  "sleep_hours_per_day": 7.5,
  "screen_time_hours_per_day": 4,
  "bmi": 26.4,
  "systolic_bp": 120,
  "diastolic_bp": 78,
  "heart_rate": 72,
  "cholesterol_total": 185,
  "hdl_cholesterol": 55,
  "triglycerides": 120,
  "gender": "Female",
  "ethnicity": "White",
  "education_level": "Graduate",
  "income_level": "Middle",
  "smoking_status": "Never",
  "employment_status": "Employed",
  "family_history_diabetes": 1,
  "hypertension_history": 0,
  "cardiovascular_history": 0
}
```

Enum options (from `api.py`):

- `gender`: `Female`, `Male`, `Other`
- `ethnicity`: `White`, `Hispanic`, `Black`, `Asian`, `Other`
- `education_level`: `Highschool`, `Graduate`, `Postgraduate`, `No formal`
- `income_level`: `Middle`, `Lower-Middle`, `Upper-Middle`, `Low`, `High`
- `smoking_status`: `Never`, `Former`, `Current`
- `employment_status`: `Employed`, `Retired`, `Unemployed`, `Student`
- Binary flags (`family_history_diabetes`, `hypertension_history`, `cardiovascular_history`): `0` or `1`

## API reference

| Endpoint | Model | Description |
| -------- | ----- | ----------- |
| `POST /diabeto/logistic` | Logistic regression (`diabeto_LR.joblib`) | Returns the logistic regression prediction for the provided features. |
| `POST /diabeto/tree` | Decision tree (`diabeto_DC.joblib`) | Returns the decision-tree prediction for the same payload. |

A successful response looks like:

```json
{
  "prediction": "Smoker"
}
```

(Again, rename these strings if you need diabetes-specific wording.)

## Custom preprocessing

`clipper.py` defines `iqr_clipper`, an `sklearn` transformer that calculates Tukey fences for specified columns during `fit` and replaces outliers with the column median during `transform`. It is baked into the serialized pipelines, so you do **not** need to call it manually during inference.

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `ModuleNotFoundError: No module named 'backend'` (during `joblib.load`) | Running Uvicorn from inside a subfolder so Python can’t import the project package layout. | Start the server from the project root (`uvicorn api:app --reload`). |
| `FileNotFoundError: diabeto_*.joblib` | Current working directory isn’t the repo root. | Move to `D:\playground-series-s5e12` (or equivalent) before launching Uvicorn. |
| `pydantic.errors` about enum values | Payload strings don’t match the allowed enum members. | Use the exact values shown in the schema above (case-sensitive). |

## Next steps

- Replace the `"Smoker"` / `"Non-Smoker"` strings with diabetes-specific labels to avoid confusion.
- Add automated tests (e.g., pytest + httpx) that cover happy-path and validation failures.
- Publish a `requirements.txt` or `pyproject.toml` so dependencies stay in sync across environments.
