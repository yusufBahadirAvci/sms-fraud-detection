# Design Patterns

| Pattern | File | Why |
|---|---|---|
| Singleton | `src/sms_fraud/config.py` (`get_config`) | The YAML file is read only once, and every module gets the same config object. |
| Pipeline | `src/sms_fraud/data.py` (`main`) | Data goes through fixed steps in order (download → load → validate → split → save), and each step's output is the next step's input. |
| Pipeline | `src/sms_fraud/train.py` (`build_pipeline`) | TF-IDF and the classifier are joined into one scikit-learn `Pipeline`, so training and prediction always run the same steps in the same order. |
| Factory | `src/sms_fraud/train.py` (`build_classifier`) | One function creates the right classifier from a name (`"logreg"` or `"nb"`), so `main()` and the tests never need to know each model's details. |
| Dependency Injection | `src/sms_fraud/api.py` (`get_model_service` + `Depends`) | Endpoints receive the loaded model from FastAPI instead of finding it themselves, so tests can swap in a small fake model. |
| Data Transfer Object (DTO) | `src/sms_fraud/schemas.py` | Pydantic classes only describe the data going in and out of the API, so the API contract and its validation live in one place. |

**Step 5a (Turkish data, monitoring): no new patterns.** `monitoring.py`, `turkish_data.py` and `text_stats.py` are plain functions that reuse existing code (`validate_data`, `evaluate`, the config Singleton); adding a pattern there would only add complexity.
