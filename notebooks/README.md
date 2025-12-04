# Notebooks Directory

This folder contains all weekly Jupyter notebooks for the Final Project:  
**Anomaly Detection in Network Traffic Using the NSL-KDD Dataset.**

Each notebook corresponds to a structured stage of the project workflow, from data preparation to final reporting and explainability.  
All notebooks are designed to be fully reproducible and rely on shared utilities in the `src/` module and artifacts stored under `data/processed/`.

---

## 📁 Notebook Structure

### **Week 01 — Data Preparation**  
**File:** `week01_data_prep.ipynb`  
- Loads raw NSL-KDD dataset  
- Cleans, preprocesses, encodes categorical variables  
- Scales continuous features  
- Saves processed artifacts (`X_train.npy`, `X_test.npy`, encoders, scalers)

---

### **Week 02 — Exploratory Data Analysis (EDA)**  
**File:** `week02_eda.ipynb`  
- Statistical summaries  
- Feature distributions  
- Correlations + heatmaps  
- PCA visualizations  
- t-SNE and UMAP (stratified sampling)  
- Saves figures to `notebooks/figures/`

---

### **Week 03 — Statistical & Unsupervised Anomaly Detection**  
**File:** `week03_anomaly_detection.ipynb`  
- Z-Score, IQR thresholding  
- Elliptic Envelope  
- Mahalanobis distance (robust covariance, chi-square thresholding)  
- DBSCAN, LOF  
- Saves detection results for use in Week 05  
- Figures stored in `figures/week03/`

---

### **Week 04 — Machine Learning–Based Anomaly Detection**  
**File:** `week04_ml_models.ipynb`  
- Train/Test split loading from Week 01 artifacts  
- Supervised models: Logistic Regression, Random Forest, SVM-RBF  
- Unsupervised ML: Isolation Forest, One-Class SVM  
- Hyperparameter tuning via GridSearchCV  
- Saves metrics + runtime tables to `reports/week04/`

---

### **Week 05 — Model Evaluation & Comparison**  
**File:** `week05_eval.ipynb`  
- Unified evaluation across all Week 03 & 04 models  
- Precision/Recall/F1 (overall + per-class)  
- ROC/PR curves  
- Runtime comparisons  
- Combined metric table summary  
- Saves final tables to `reports/week05/`

---

### **Week 06 — Explainability & Final Report**  
**File:** `week06_report_explain.ipynb`  
- SHAP explainability for supervised models (RF, SVM)  
- Permutation importance  
- AE explainability policy (no SHAP for AE)  
- IDS implications + narrative summary  
- Figures shown in-grid and saved  
- Final report elements for Quarto manuscript

---

## 📦 Folder Conventions
```text
notebooks/
│
├── figures/
│ ├── week02/
│ ├── week03/
│ ├── week04/
│ └── week06/
│
├── week01_data_prep.ipynb
├── week02_eda.ipynb
├── week03_anomaly_detection.ipynb
├── week04_ml_models.ipynb
├── week05_eval.ipynb
└── week06_report_explain.ipynb
```
