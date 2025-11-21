# Source Code (`src/`)

This folder contains all **preprocessing** and **utility** scripts used in the project  
**“Anomaly Detection in Network Traffic Using the NSL-KDD Dataset.”**

### Module Descriptions

- **eval.py** — Evaluation utilities: metrics, confusion matrices, ROC/F1 reporting, comparison helpers.  
- **explain.py** — SHAP explainability, feature importance, PCA decision boundaries, model interpretation tools.  
- **io.py** — File loading/saving helpers for CSV, Parquet, NumPy, and model artifacts.  
- **models.py** — Supervised models (LR, SVM, RF, XGBoost, Autoencoder wrappers) and training functions.  
- **plots.py** — Visualization helpers for distributions, correlation heatmaps, ROC curves, and reconstruction error plots.  
- **preps.py** — Preprocessing utilities: encoding, scaling, cleaning, feature engineering, and dataset preparation.  
- **unsupervised.py** — Unsupervised anomaly detectors (Z-score, Elliptic Envelope, LOF, Isolation Forest, Mahalanobis).  
- **thresholds.py** — Thresholding logic for anomaly scoring, chi-square cutoffs, and scoring transformations.  
- **utils.py** — General utilities, helper functions, logging, and miscellaneous shared functionality.  

```text
src/
├── eval.py
├── explain.py
├── io.py
├── models.py
├── plots.py
├── preps.py
├── unsupervised.py
├── thresholds.py
└── utils.py


