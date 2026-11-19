# Design Patterns

| Pattern | File | Why |
|---|---|---|
| Singleton | `src/sms_fraud/config.py` (`get_config`) | The YAML file is read only once, and every module gets the same config object. |
| Pipeline | `src/sms_fraud/data.py` (`main`) | Data goes through fixed steps in order (download → load → validate → split → save), and each step's output is the next step's input. |
| Pipeline | `src/sms_fraud/train.py` (`build_pipeline`) | TF-IDF and the classifier are joined into one scikit-learn `Pipeline`, so training and prediction always run the same steps in the same order. |
| Factory | `src/sms_fraud/train.py` (`build_classifier`) | One function creates the right classifier from a name (`"logreg"` or `"nb"`), so `main()` and the tests never need to know each model's details. |
| Factory | `src/sms_fraud/train.py` (`build_features`) | One function creates the TF-IDF step from a name (`"word"`, `"char"`, `"word+char"`), so a new experiment is one line in `config.yaml`, not new code. |
| Strategy | `src/sms_fraud/promote.py` (`decide`) | The promotion rule is one small pure function with its limit passed in, so it is tested with fake metrics and can change without touching the MLflow and file code. |
| Dependency Injection | `src/sms_fraud/api.py` (`get_model_service` + `Depends`) | Endpoints receive the loaded model from FastAPI instead of finding it themselves, so tests can swap in a small fake model. |
| Test Double (Dummy) | `scripts/make_dummy_model.py` (`build_dummy_model`) | A tiny model with the same shape as the real one (a Pipeline with a `"tfidf"` step), so CI can build and start the API image without the real model, which exists only in the local DVC cache. |
| Data Transfer Object (DTO) | `src/sms_fraud/schemas.py` | Pydantic classes only describe the data going in and out of the API, so the API contract and its validation live in one place. |

**Step 5a (Turkish data, monitoring): no new patterns.** `monitoring.py`, `turkish_data.py` and `text_stats.py` are plain functions that reuse existing code (`validate_data`, `evaluate`, the config Singleton); adding a pattern there would only add complexity.

**Step 5b (Turkish training, champion/challenger):** two new entries above (`build_features` Factory, `decide` Strategy). The existing Pipeline pattern in `build_pipeline` now also covers the `FeatureUnion` (word + char TF-IDF side by side), so no new pattern was needed there. Champion/challenger itself is an MLOps practice built from plain functions and MLflow aliases, not a separate class pattern.

**Step 6a (CI with GitHub Actions):** one new entry above (Test Double in `make_dummy_model.py`). The new smoke test (`tests/test_pipeline_smoke.py`) reuses the config Singleton: it points `PROJECT_ROOT` to a temporary folder, and every stage follows, because all paths come from `project_path()`. No other pattern was added; the CI workflow is configuration, not code.
