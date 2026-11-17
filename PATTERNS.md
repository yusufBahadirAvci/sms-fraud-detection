# Design Patterns

| Pattern | File | Why |
|---|---|---|
| Singleton | `src/sms_fraud/config.py` (`get_config`) | The YAML file is read only once, and every module gets the same config object. |
| Pipeline | `src/sms_fraud/data.py` (`main`) | Data goes through fixed steps in order (download → load → validate → split → save), and each step's output is the next step's input. |
| Pipeline | `src/sms_fraud/train.py` (`build_pipeline`) | TF-IDF and the classifier are joined into one scikit-learn `Pipeline`, so training and prediction always run the same steps in the same order. |
| Factory | `src/sms_fraud/train.py` (`build_classifier`) | One function creates the right classifier from a name (`"logreg"` or `"nb"`), so `main()` and the tests never need to know each model's details. |
