# Concepts Study Guide (Exam Preparation)

Every concept used in this project, grouped by project stage.
For each concept you get:

- **Definition**: simple English
- **Türkçe**: short explanation in Turkish
- **In this project**: the file or command where you can see it
- **Exam Q / A**: one likely exam question and a short model answer

Tip: learn the **Exam A** sentences by heart. They are short and use the right words.

## Table of contents

1. [Big picture](#1-big-picture)
   - [ML lifecycle](#ml-lifecycle) · [MLOps](#mlops)
2. [Data](#2-data)
   - [Data validation](#data-validation) · [Imbalanced data](#imbalanced-data) ·
     [Data leakage](#data-leakage) · [Stratified split](#stratified-split) ·
     [Group split](#group-split)
3. [Data and pipeline versioning](#3-data-and-pipeline-versioning)
   - [DVC and remotes](#dvc-and-remotes) · [Pipeline reproducibility](#pipeline-reproducibility) ·
     [Idempotency](#idempotency)
4. [Model training](#4-model-training)
   - [TF-IDF: word vs char n-grams](#tf-idf-word-vs-char-n-grams) · [Overfitting](#overfitting)
5. [Evaluation](#5-evaluation)
   - [Precision, recall, F1](#precision-recall-f1) · [Challenge set](#challenge-set)
6. [Experiments and model registry](#6-experiments-and-model-registry)
   - [Experiment tracking (MLflow)](#experiment-tracking-mlflow) ·
     [Model registry and aliases](#model-registry-and-aliases) ·
     [Champion / challenger](#champion--challenger) · [Rollback](#rollback)
7. [Serving the model](#7-serving-the-model)
   - [FastAPI](#fastapi) · [Pydantic](#pydantic) · [Docker](#docker) ·
     [Healthcheck](#healthcheck) · [Docker Compose](#docker-compose)
8. [Automation (CI/CD)](#8-automation-cicd)
   - [CI](#ci-continuous-integration) · [CD](#cd-continuous-delivery) ·
     [GitHub Actions](#github-actions) · [Secrets](#secrets) · [Coverage](#coverage) ·
     [Linting](#linting)
9. [Monitoring](#9-monitoring)
   - [Data drift](#data-drift) · [Concept drift](#concept-drift) · [OOV](#oov-out-of-vocabulary) ·
     [KS test](#ks-test) · [PSI](#psi-population-stability-index) ·
     [Monitoring and alerts](#monitoring-and-alerts)
10. [Design patterns](#10-design-patterns)
    - [Singleton](#singleton) · [Pipeline (pattern)](#pipeline-pattern) · [Factory](#factory) ·
      [Strategy](#strategy) · [Dependency Injection](#dependency-injection) ·
      [Test Double](#test-double-dummy) · [Immutable Artifact](#immutable-artifact) ·
      [Idempotent Receiver](#idempotent-receiver) · [Repeatable Splitting](#repeatable-splitting) ·
      [Partitioning](#partitioning) · [Data Transfer Object](#data-transfer-object-dto)

---

## 1. Big picture

### ML lifecycle

- **Definition:** The steps of a machine learning project: get data, prepare it, train a model,
  evaluate it, deploy it, and monitor it. Then the loop starts again.
- **Türkçe:** Bir makine öğrenmesi projesinin adımları: veri toplama, hazırlama, eğitim,
  değerlendirme, yayına alma ve izleme. Sonra döngü yeniden başlar.
- **In this project:** `dvc.yaml` (prepare → train → promote → challenge → monitor),
  then `api.py` (deploy) and the weekly `monitor` job in `.github/workflows/cd.yml`.
- **Exam Q:** Why is the ML lifecycle a loop and not a straight line?
- **Exam A:** Because data in the real world changes. Monitoring finds problems, and then we
  collect new data and retrain the model.

### MLOps

- **Definition:** MLOps means using DevOps ideas (automation, versioning, testing, monitoring)
  for machine learning. The goal is to put models into production safely and repeatably.
- **Türkçe:** MLOps, DevOps fikirlerini (otomasyon, sürümleme, test, izleme) makine öğrenmesine
  uygulamaktır. Amaç, modeli güvenli ve tekrarlanabilir şekilde canlıya almaktır.
- **In this project:** The whole repo: DVC (data versions), MLflow (experiments), Docker,
  GitHub Actions (CI/CD), monitoring.
- **Exam Q:** What is the difference between DevOps and MLOps?
- **Exam A:** DevOps versions and tests code. MLOps must also version data and models, track
  experiments, and monitor model quality, because a model can get worse even if the code does not change.

---

## 2. Data

### Data validation

- **Definition:** Checking that the data is correct before we use it. For example: no empty
  rows, only allowed labels, no duplicates.
- **Türkçe:** Veriyi kullanmadan önce doğru olup olmadığını kontrol etmek. Örneğin boş satır
  olmaması, sadece izin verilen etiketler, tekrar eden satır olmaması.
- **In this project:** `src/sms_fraud/data.py` → `validate_data()` (empty messages, labels
  `ham`/`spam`, duplicates, HTML codes like `&lt;`). The API also validates input with Pydantic.
- **Exam Q:** Why do we validate data before training?
- **Exam A:** Bad data gives a bad model ("garbage in, garbage out"). Validation finds errors early
  and logs how many rows were removed, so the problem is visible.

### Imbalanced data

- **Definition:** One class has many more examples than the other. In English SMS, only about
  12% of messages are spam.
- **Türkçe:** Bir sınıfın örnek sayısı diğerinden çok daha fazladır. İngilizce SMS verisinde
  mesajların sadece yaklaşık %12'si spam.
- **In this project:** English data (12.4% spam); `eda.py` shows the class balance. We use
  F1 instead of accuracy, and a stratified split.
- **Exam Q:** Why is accuracy a bad metric for imbalanced data?
- **Exam A:** A model that says "ham" for every message gets about 88% accuracy here, but it finds
  zero spam. Precision, recall and F1 show this problem; accuracy hides it.

### Data leakage

- **Definition:** Information from the test set gets into training. Then the test score is too
  high and does not show real performance.
- **Türkçe:** Test verisinden bilgi eğitime sızar. Bu durumda test skoru olduğundan yüksek
  görünür ve gerçek performansı göstermez.
- **In this project:** `src/sms_fraud/leakage.py`, report `reports/leakage.md`. With the old
  random split, 5.1% (English) and 11.4% (Turkish) of test messages had a near-copy in train.
  Command: `dvc repro leakage`.
- **Exam Q:** How did you find leakage in your project, and how did you fix it?
- **Exam A:** I normalized the text and compared every test message to all train messages with
  char TF-IDF cosine similarity. Pairs with similarity ≥ 0.9 were near-duplicates. I fixed it with a
  group split: all near-copies go to the same side.

### Stratified split

- **Definition:** A train/test split that keeps the class ratio the same in both parts
  (for example 12% spam in train and 12% spam in test).
- **Türkçe:** Sınıf oranını eğitim ve test kümelerinde aynı tutan bölme yöntemi
  (örneğin her ikisinde de %12 spam).
- **In this project:** `data.py` → `split_data()` (`stratify=df["label"]` and
  `StratifiedGroupKFold`).
- **Exam Q:** Why do we use a stratified split?
- **Exam A:** With a small spam class, a random split could put too few spam messages in the test set.
  Stratification keeps the same spam share, so the test score is more stable.

### Group split

- **Definition:** A split where all rows of one group go to the same side (train OR test, never both).
- **Türkçe:** Bir gruptaki tüm satırların aynı tarafa (ya eğitim ya test) gittiği bölme yöntemi.
- **In this project:** `data.py` → `leakage_safe_split()`. Groups are near-duplicate clusters from
  `leakage.cluster_ids()`. The split uses `StratifiedGroupKFold` (group + stratified).
- **Exam Q:** When do you need a group split instead of a random split?
- **Exam A:** When rows are not independent, for example many copies of the same spam template.
  A random split puts copies in train and test, which is leakage. A group split prevents it.

---

## 3. Data and pipeline versioning

### DVC and remotes

- **Definition:** DVC (Data Version Control) versions big files (data, models) next to Git.
  Git stores only small `.dvc` pointer files; the real files go to a cache and a **remote** storage.
- **Türkçe:** DVC, büyük dosyaları (veri, model) Git ile birlikte sürümler. Git sadece küçük
  `.dvc` dosyalarını tutar; gerçek dosyalar önbelleğe ve uzak depoya (remote) gider.
- **In this project:** `dvc.yaml`, `dvc.lock`, `models/model.joblib.dvc`, remote `origin` on
  DagsHub (`.dvc/config`). Commands: `dvc pull`, `dvc push`, `dvc status -c`.
- **Exam Q:** Why not put the data and model directly into Git?
- **Exam A:** Git is made for small text files. Big binary files make the repo slow and heavy.
  DVC keeps them in a remote and Git keeps only their hash, so we still know which data belongs to which commit.

### Pipeline reproducibility

- **Definition:** Anyone can run the same pipeline with the same inputs and get the same results.
- **Türkçe:** Herkes aynı girdilerle aynı adımları çalıştırıp aynı sonuçları alabilir.
- **In this project:** `dvc.yaml` (stages with deps and outs), `dvc.lock` (hashes),
  fixed seed `random_state: 42` in `config/config.yaml`, pinned versions in `requirements*.txt`,
  LF line endings in `.gitattributes`. Command: `dvc repro`.
- **Exam Q:** Name three things that make your results reproducible.
- **Exam A:** A fixed random seed, pinned package versions, and a DVC pipeline whose lock file records
  the hash of every input and output.

### Idempotency

- **Definition:** An operation is idempotent if running it many times gives the same result as
  running it once.
- **Türkçe:** Bir işlem birçok kez çalıştırıldığında, bir kez çalıştırılmasıyla aynı sonucu
  veriyorsa idempotenttir.
- **In this project:** `dvc repro` (runs only changed stages; a second run does nothing),
  the alert step in `cd.yml` (comments on the open issue instead of opening a new one),
  `mkdir(exist_ok=True)` in the report writers.
- **Exam Q:** Why is idempotency important in automation?
- **Exam A:** Jobs can fail and be retried. If the job is idempotent, a retry does not create
  duplicates or a different result, so retries are safe.

---

## 4. Model training

### TF-IDF: word vs char n-grams

- **Definition:** TF-IDF turns text into numbers. A term gets a high weight if it is frequent in
  this message but rare in other messages. **Word n-grams** use words and word pairs;
  **char n-grams** use pieces of 3–5 letters.
- **Türkçe:** TF-IDF metni sayılara çevirir. Bir terim bu mesajda sık, diğer mesajlarda nadir ise
  yüksek ağırlık alır. Kelime n-gram'ları kelimeleri, karakter n-gram'ları 3–5 harflik parçaları kullanır.
- **In this project:** `train.py` → `build_features()`. Runs: `A_word`, `B_char` (champion),
  `C_word_char` in `config/config.yaml`.
- **Exam Q:** Why do char n-grams work better for Turkish?
- **Exam A:** Turkish adds many suffixes to one word ("tıkla", "tıklayın", "tıklayınız"). With words,
  each form is a new unknown word. With letter pieces, the forms share most pieces, so the model
  still recognizes them.

### Overfitting

- **Definition:** The model learns the training data too well, also its noise. It scores high on
  train data but worse on new data.
- **Türkçe:** Model eğitim verisini, gürültüsüyle birlikte fazla iyi öğrenir. Eğitimde yüksek,
  yeni veride düşük başarı gösterir.
- **In this project:** We test on a separate clean test set and on a challenge set
  (`reports/challenge.md`). `C` in `config.yaml` controls how closely logistic regression fits.
  Turkish F1 0.998 vs challenge recall 0.85 shows the model does not generalize perfectly to new kinds of fraud.
- **Exam Q:** How can you detect overfitting?
- **Exam A:** Compare scores on training data and on unseen data. A big gap means overfitting.
  A challenge set with new kinds of messages is an even stronger test.

---

## 5. Evaluation

### Precision, recall, F1

- **Definition:** **Precision** = of the messages we called spam, how many were really spam.
  **Recall** = of the real spam messages, how many we found. **F1** = the harmonic mean of the two.
- **Türkçe:** Precision: spam dediğimiz mesajların kaçı gerçekten spam. Recall: gerçek spam
  mesajlarının kaçını bulduk. F1: ikisinin harmonik ortalaması.
- **In this project:** `train.py` → `evaluate()`; `reports/metrics.json`. Champion: English
  precision 0.983, recall 0.883, F1 0.930.
- **Exam Q:** For SMS fraud, is precision or recall more important?
- **Exam A:** Both matter. Low recall means fraud reaches users. Low precision means we block normal
  messages. F1 balances them; the threshold (0.5 in `config.yaml`) can move the balance.

### Challenge set

- **Definition:** A small, hand-made test set with hard or new examples. It is used **only for
  evaluation**, never for training or tuning.
- **Türkçe:** Zor veya yeni örneklerden oluşan, elle hazırlanmış küçük bir test kümesi. Sadece
  değerlendirme için kullanılır, eğitimde asla kullanılmaz.
- **In this project:** `data/challenge/tr_phishing.csv` (40 Turkish messages), `challenge.py`,
  DVC stage `challenge`, report `reports/challenge.md` (recall 0.85: 3 of 20 phishing messages missed).
- **Exam Q:** Why did you add a challenge set if you already have a test set?
- **Exam A:** The test set comes from the same source as the training data, so it is "easy". The
  challenge set has modern phishing that the dataset does not have. It shows the real weakness of the model.

---

## 6. Experiments and model registry

### Experiment tracking (MLflow)

- **Definition:** Saving every training run with its parameters, metrics and files, so we can
  compare runs later.
- **Türkçe:** Her eğitim çalıştırmasını parametreleri, metrikleri ve dosyalarıyla kaydetmek;
  böylece çalıştırmaları sonra karşılaştırabiliriz.
- **In this project:** `train.py` → `run_experiment()` (`mlflow.log_params`, `log_metrics`,
  `log_artifact`). Store: `mlruns/mlflow.db`. UI: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Exam Q:** What do you log in MLflow for one run?
- **Exam A:** Parameters (features, C, n-gram range), metrics (en/tr precision, recall, F1),
  tags (split, step), artifacts (confusion matrices) and the model itself.

### Model registry and aliases

- **Definition:** A model registry stores model versions (v1, v2, ...). An **alias** is a named
  pointer to one version, like `champion`. Moving an alias is fast and easy to undo.
- **Türkçe:** Model kayıt defteri model sürümlerini (v1, v2, ...) saklar. Alias, bir sürümü gösteren
  isimli bir işaretçidir (ör. `champion`). Alias'ı taşımak hızlı ve geri alınabilir.
- **In this project:** Registered model `sms-fraud-classifier`. Aliases `champion` (v19) and
  `previous` (v13). Code: `promote.py` → `move_aliases()`.
- **Exam Q:** Why use aliases instead of version numbers in the code?
- **Exam A:** The code asks for "champion" and does not change. To deploy another model we only move the
  alias. Rollback is also only moving an alias.

### Champion / challenger

- **Definition:** The **champion** is the model in production. A **challenger** is a new model.
  The challenger replaces the champion only if it passes clear rules.
- **Türkçe:** Champion, canlıdaki modeldir. Challenger yeni modeldir. Challenger, sadece belirli
  kuralları geçerse champion'ın yerini alır.
- **In this project:** `promote.py` → `decide()`. Rules: Turkish F1 must be higher, English F1 may drop
  at most 0.02. Report: `reports/promotion.md`. In v1.0.0 the old champion was scored only on rows it
  never saw (`--champion-split old`), so the contest was fair.
- **Exam Q:** Why compare both models on the same test data?
- **Exam A:** Otherwise the comparison is not fair. Our old champion had seen 80% of the new test
  rows in training, so its scores were too high. We scored it only on unseen rows.

### Rollback

- **Definition:** Going back to the previous working model (or version) when the new one has problems.
- **Türkçe:** Yeni model sorun çıkarırsa önceki çalışan modele (veya sürüme) geri dönmek.
- **In this project:** `python -m src.sms_fraud.promote rollback` (swaps `champion` and `previous`
  aliases and restores `models/model.joblib`). Docker: run an older image sha tag from GHCR.
- **Exam Q:** How do you roll back a bad model in your project?
- **Exam A:** I run `promote rollback`. It loads the "previous" version from MLflow, writes it to
  `models/model.joblib` and swaps the aliases. Then `dvc add` and redeploy. With Docker, I can also
  run the older image tag.

---

## 7. Serving the model

### FastAPI

- **Definition:** A Python web framework for building REST APIs. It creates interactive docs
  (`/docs`) automatically.
- **Türkçe:** REST API geliştirmek için bir Python web çatısı. `/docs` sayfasını otomatik oluşturur.
- **In this project:** `src/sms_fraud/api.py`: `GET /health`, `POST /predict`, `POST /predict/batch`.
  Run: `uvicorn src.sms_fraud.api:app --port 8000`.
- **Exam Q:** Why does the API load the model once at startup and not on every request?
- **Exam A:** Loading a model is slow. Loading it once (lifespan) makes every request fast.

### Pydantic

- **Definition:** A Python library that checks data types and rules with classes. Wrong input
  gives a clear error.
- **Türkçe:** Veri tiplerini ve kurallarını sınıflarla kontrol eden bir Python kütüphanesi.
  Hatalı girdi açık bir hata verir.
- **In this project:** `src/sms_fraud/schemas.py` (`PredictRequest`, `PredictResponse`, ...).
  An empty message gives HTTP 422.
- **Exam Q:** What happens if a client sends an empty message?
- **Exam A:** The Pydantic validator raises an error, and FastAPI answers with HTTP 422
  (Unprocessable Entity). The model is never called.

### Docker

- **Definition:** Docker packages an app with everything it needs (Python, libraries, model) into an
  **image**. A running image is a **container**. It runs the same on every computer.
- **Türkçe:** Docker, uygulamayı ihtiyaç duyduğu her şeyle (Python, kütüphaneler, model) bir imaj
  içinde paketler. Çalışan imaja konteyner denir. Her bilgisayarda aynı çalışır.
- **In this project:** `Dockerfile.api`, `Dockerfile.ui`, `.dockerignore`. Images on GHCR.
  Non-root user `appuser`.
- **Exam Q:** Why does the container run as a non-root user?
- **Exam A:** Security. If someone breaks into the app, they do not get root rights in the container.

### Healthcheck

- **Definition:** A small test that Docker runs again and again to see if the app really works.
  The container becomes `healthy` or `unhealthy`.
- **Türkçe:** Docker'ın uygulamanın gerçekten çalışıp çalışmadığını görmek için tekrar tekrar
  yaptığı küçük bir test. Konteyner `healthy` veya `unhealthy` olur.
- **In this project:** `HEALTHCHECK` in `Dockerfile.api` (calls `/health`) and `Dockerfile.ui`.
  CI and CD wait for `healthy`.
- **Exam Q:** What is the difference between "running" and "healthy"?
- **Exam A:** "Running" only means the process started. "Healthy" means the app answers correctly,
  for example the model is loaded.

### Docker Compose

- **Definition:** A tool to start several containers together from one YAML file.
- **Türkçe:** Birden fazla konteyneri tek bir YAML dosyasından birlikte başlatan araç.
- **In this project:** `docker-compose.yml` (services `api` and `ui`; `ui` waits for
  `service_healthy`; `./logs` volume). Command: `docker compose up --build -d`.
- **Exam Q:** How does the UI container find the API container?
- **Exam A:** Compose puts both in one network. The UI uses the service name: `API_URL=http://api:8000`.
  `localhost` would mean the UI container itself.

---

## 8. Automation (CI/CD)

### CI (Continuous Integration)

- **Definition:** Every push is automatically checked: lint, tests, build. Problems are found before
  they reach users.
- **Türkçe:** Her push otomatik olarak kontrol edilir: lint, testler, build. Sorunlar kullanıcıya
  ulaşmadan bulunur.
- **In this project:** `.github/workflows/ci.yml`: job `test` (ruff, pytest with coverage) and job
  `docker` (build images, start API with a dummy model, call `/health` and `/predict`).
- **Exam Q:** Why does CI use a dummy model?
- **Exam A:** CI should be fast and need no secrets (pull requests from forks get no secrets).
  The real model is in the DVC remote. A dummy model with the same shape lets CI test the Docker
  image without it. CD later builds the image with the real model.

### CD (Continuous Delivery)

- **Definition:** After CI passes, the new version is automatically built and delivered (for example
  as a Docker image in a registry).
- **Türkçe:** CI geçtikten sonra yeni sürüm otomatik olarak build edilir ve teslim edilir
  (ör. registry'de bir Docker imajı olarak).
- **In this project:** `.github/workflows/cd.yml`, job `deliver`: pulls the real model from DagsHub,
  builds images, smoke-tests them, pushes to GHCR with tags `<sha>` and `latest`.
- **Exam Q:** What is the difference between continuous delivery and continuous deployment?
- **Exam A:** Delivery: the release is ready automatically (image in a registry), but a human starts it.
  Deployment: it also goes to production automatically. We do continuous delivery.

### GitHub Actions

- **Definition:** GitHub's automation service. Workflows are YAML files that run jobs on events
  (push, schedule, button).
- **Türkçe:** GitHub'ın otomasyon servisi. İş akışları, olaylarla (push, zamanlama, buton) tetiklenen
  YAML dosyalarıdır.
- **In this project:** `.github/workflows/ci.yml` (on push / pull request), `cd.yml` (on CI success,
  every Monday by `cron`, and by button with `workflow_dispatch`).
- **Exam Q:** How does your CD workflow know that CI passed?
- **Exam A:** It uses the `workflow_run` event and checks `conclusion == 'success'`. A failed CI never delivers.

### Secrets

- **Definition:** Passwords and tokens. They must never be in the code or in Git.
- **Türkçe:** Şifreler ve token'lar. Asla kodda veya Git'te olmamalıdır.
- **In this project:** `DAGSHUB_TOKEN` in GitHub Settings → Secrets; locally in `.dvc/config.local`
  (git-ignored). `GITHUB_TOKEN` for GHCR. Workflows use least privilege (`permissions:`).
- **Exam Q:** How do you give the DagsHub token to the CD job safely?
- **Exam A:** It is a GitHub Actions secret. The job writes it only to `.dvc/config.local` in the runner.
  GitHub hides it in logs, and the runner is deleted after the job.

### Coverage

- **Definition:** The share of code lines that the tests run. It shows which code is not tested.
- **Türkçe:** Testlerin çalıştırdığı kod satırlarının oranı. Hangi kodun test edilmediğini gösterir.
- **In this project:** `pytest --cov=src --cov-fail-under=70` in `ci.yml`; `coverage.xml` is uploaded.
- **Exam Q:** Does 100% coverage mean the code has no bugs?
- **Exam A:** No. It only means each line ran once. The tests can still miss wrong results or edge cases.

### Linting

- **Definition:** Automatic checking of code style and simple errors (unused imports, wrong order)
  without running the code.
- **Türkçe:** Kodu çalıştırmadan stil ve basit hataları (kullanılmayan import, yanlış sıra)
  otomatik kontrol etmek.
- **In this project:** `ruff check .` and `ruff format --check .` (rules in `pyproject.toml`), first step of CI.
- **Exam Q:** Why run the linter in CI and not only on your computer?
- **Exam A:** CI checks every push the same way. So no one can forget it, and the code style stays the same for everyone.

---

## 9. Monitoring

### Data drift

- **Definition:** The input data in production changes compared to the training data (for example
  new words, longer messages, a new language).
- **Türkçe:** Canlıdaki girdi verisi, eğitim verisine göre değişir (ör. yeni kelimeler, daha uzun
  mesajlar, yeni bir dil).
- **In this project:** `src/sms_fraud/monitoring.py` checks OOV ratio, length (KS test) and spam
  probability (PSI). Report: `reports/monitoring/drift_report.md`.
- **Exam Q:** How do you detect data drift without labels?
- **Exam A:** Compare input statistics between a reference window and the current window: unknown
  word share (OOV), message length (KS test), and the model's probability distribution (PSI).

### Concept drift

- **Definition:** The relation between input and label changes. The same kind of message now means
  something else (for example new phishing tricks that look like normal service SMS).
- **Türkçe:** Girdi ile etiket arasındaki ilişki değişir. Aynı tür mesaj artık başka bir anlam taşır
  (ör. normal servis SMS'ine benzeyen yeni oltalama yöntemleri).
- **In this project:** The F1-drop check in `monitoring.py` (needs labels), and the challenge set
  (new phishing types the model misses).
- **Exam Q:** What is the difference between data drift and concept drift?
- **Exam A:** Data drift: the inputs change. Concept drift: the meaning (input → label) changes.
  Concept drift usually needs labels to detect, for example a drop in F1.

### OOV (Out-of-vocabulary)

- **Definition:** Words the model never saw in training. The OOV ratio is the share of such words
  in a message.
- **Türkçe:** Modelin eğitimde hiç görmediği kelimeler. OOV oranı, bir mesajdaki bu kelimelerin payıdır.
- **In this project:** `text_stats.py` → `oov_ratio()`; threshold 0.5 in `config.yaml`; logged by the API.
- **Exam Q:** Why is a high OOV ratio a warning sign?
- **Exam A:** The model knows nothing about these words, so its prediction is based on little
  information. In Step 5a, Turkish messages had a high OOV ratio for the English model.

### KS test

- **Definition:** The Kolmogorov–Smirnov test checks if two samples come from the same distribution.
  A small p-value (< 0.05) means they are really different.
- **Türkçe:** Kolmogorov–Smirnov testi, iki örneğin aynı dağılımdan gelip gelmediğini kontrol eder.
  Küçük p-değeri (< 0.05) gerçekten farklı oldukları anlamına gelir.
- **In this project:** `monitoring.py` → `ks_2samp(ref["length"], cur["length"])` on message length.
- **Exam Q:** What does a KS p-value of 0.70 mean?
- **Exam A:** The two length distributions are very likely the same. There is no evidence of drift.

### PSI (Population Stability Index)

- **Definition:** A number that shows how much a distribution moved. We cut values into bins and
  compare the share in each bin. PSI > 0.2 is usually a big shift.
- **Türkçe:** Bir dağılımın ne kadar kaydığını gösteren sayı. Değerler aralıklara bölünür ve her
  aralıktaki pay karşılaştırılır. PSI > 0.2 genellikle büyük bir kaymadır.
- **In this project:** `monitoring.py` → `psi()` on the spam probability (10 bins, smoothing for empty bins).
- **Exam Q:** Why calculate PSI on the predicted probability?
- **Exam A:** It needs no labels, and it shows if the model behaves differently on the new data,
  for example if it suddenly predicts much more spam.

### Monitoring and alerts

- **Definition:** Checking a model in production regularly, and sending an alert when a value crosses
  a threshold.
- **Türkçe:** Canlıdaki modeli düzenli kontrol etmek ve bir değer eşiği aşınca uyarı göndermek.
- **In this project:** Per-language windows (`compare_by_language`), `alert.json`, the weekly `monitor`
  job in `cd.yml` that opens a GitHub issue. Issue #1 came from mixing English and Turkish in one window.
- **Exam Q:** Why did you split monitoring by language?
- **Exam A:** With one window, a change in the language mix looked like drift (false alerts). With
  one reference per language, we compare English with English and Turkish with Turkish.

---

## 10. Design patterns

### Singleton

- **Definition:** Only one object of a kind exists, and everyone uses that same object.
- **Türkçe:** Bir türden sadece bir nesne vardır ve herkes aynı nesneyi kullanır.
- **In this project:** `src/sms_fraud/config.py` → `get_config()`: the YAML file is read once.
- **Exam Q:** Why is the config a Singleton?
- **Exam A:** The file is read only once, and all modules see the same settings.

### Pipeline (pattern)

- **Definition:** Data goes through fixed steps in order; the output of one step is the input of the next.
- **Türkçe:** Veri sabit adımlardan sırayla geçer; bir adımın çıktısı bir sonrakinin girdisidir.
- **In this project:** `data.py` → `main()` (download → load → validate → split → save) and
  `train.py` → `build_pipeline()` (scikit-learn `Pipeline`: TF-IDF → classifier).
- **Exam Q:** Why put TF-IDF and the classifier in one scikit-learn Pipeline?
- **Exam A:** Training and prediction always run the same steps. The API cannot forget the TF-IDF step,
  and we save only one file.

### Factory

- **Definition:** A function that creates the right object from a name, so the caller does not need the details.
- **Türkçe:** İsme göre doğru nesneyi oluşturan bir fonksiyon; çağıran kodun detayları bilmesine gerek kalmaz.
- **In this project:** `train.py` → `build_classifier()` (`"logreg"`, `"nb"`) and `build_features()`
  (`"word"`, `"char"`, `"word+char"`).
- **Exam Q:** How does the Factory help when you add a new experiment?
- **Exam A:** A new experiment is one line in `config.yaml`. The training code does not change.

### Strategy

- **Definition:** A behaviour (an algorithm or rule) is put into its own small, replaceable unit.
- **Türkçe:** Bir davranış (algoritma veya kural) kendi küçük, değiştirilebilir birimine konur.
- **In this project:** `promote.py` → `decide()` (promotion rule) and `text_stats.py` →
  `detect_language()` (language detector).
- **Exam Q:** What is the benefit of `decide()` being a pure function?
- **Exam A:** It can be tested with fake numbers, without MLflow or files, and the rule can change in one place.

### Dependency Injection

- **Definition:** An object gets the things it needs from outside, instead of creating them itself.
- **Türkçe:** Bir nesne ihtiyaç duyduğu şeyleri kendisi oluşturmak yerine dışarıdan alır.
- **In this project:** `api.py` → `get_model_service()` with FastAPI `Depends`.
- **Exam Q:** How does Dependency Injection help testing?
- **Exam A:** Endpoints do not load the model themselves; they receive it. So in tests
  (`tests/test_api.py`) we give the app a small fake model instead. No real model file is needed.

### Test Double (Dummy)

- **Definition:** A fake object that replaces a real one in tests. It has the same shape but is simple.
- **Türkçe:** Testlerde gerçek nesnenin yerine geçen sahte nesne. Aynı yapıya sahiptir ama basittir.
- **In this project:** `scripts/make_dummy_model.py` → `build_dummy_model()`, used by CI to build the API image.
- **Exam Q:** Why does the dummy model need a `"tfidf"` step?
- **Exam A:** The API reads the vectorizer from the `"tfidf"` step for the OOV ratio. Same shape means
  the API code works without changes.

### Immutable Artifact

- **Definition:** A build result that never changes after it is created. It has a fixed, unique name.
- **Türkçe:** Oluşturulduktan sonra hiç değişmeyen, sabit ve benzersiz isimli bir build çıktısı.
- **In this project:** `cd.yml`, job `deliver`: every image is tagged with the commit sha;
  `latest` is only a moving pointer.
- **Exam Q:** Why tag images with the commit sha?
- **Exam A:** We always know which code and model are inside, and we can roll back by running an old tag.

### Idempotent Receiver

- **Definition:** A receiver that handles the same message many times without side effects being repeated.
- **Türkçe:** Aynı mesajı birçok kez alsa bile yan etkiyi tekrar etmeyen alıcı.
- **In this project:** `cd.yml`, job `monitor`: if an open "Drift alert" issue exists, it adds a comment
  instead of opening a new issue.
- **Exam Q:** What would happen without this pattern?
- **Exam A:** Every Monday with drift would open a new issue, and we would get many duplicate issues.

### Repeatable Splitting

- **Definition:** The train/test split is always the same, because it uses a fixed seed and stable groups.
- **Türkçe:** Sabit bir seed ve sabit gruplar kullanıldığı için eğitim/test bölmesi her zaman aynıdır.
- **In this project:** `data.py` → `leakage_safe_split()` (clusters + `random_state: 42`).
- **Exam Q:** Why must the split be repeatable?
- **Exam A:** If the split changes, test scores of different runs cannot be compared, and test rows could move into training.

### Partitioning

- **Definition:** Data is divided into parts (partitions), and the same logic runs on each part separately.
- **Türkçe:** Veri parçalara bölünür ve aynı mantık her parça üzerinde ayrı ayrı çalışır.
- **In this project:** `monitoring.py` → `compare_by_language()` (one reference + current window per language).
- **Exam Q:** Which problem did Partitioning solve in monitoring?
- **Exam A:** A change in the language mix looked like drift. Now each language is compared only with itself.

### Data Transfer Object (DTO)

- **Definition:** A simple class that only carries data between parts of a system. It has no business logic.
- **Türkçe:** Sistemin parçaları arasında sadece veri taşıyan basit bir sınıf. İş mantığı içermez.
- **In this project:** `src/sms_fraud/schemas.py` (Pydantic classes for API input and output).
- **Exam Q:** Why keep the API schemas in a separate file?
- **Exam A:** The API contract and its validation are in one place. Clients and tests can see the exact input and output format.
