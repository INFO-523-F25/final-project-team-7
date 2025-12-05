[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wojP3-_r)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=21235589)
# Final Project — Anomaly Detection in Network Traffic (NSL-KDD)

## Overview
This project investigates **anomaly detection in network traffic** using the **NSL-KDD dataset**, a widely used benchmark for evaluating Intrusion Detection Systems (IDS).  
The goal is to design a **robust, explainable, and reproducible pipeline** that integrates:

- Exploratory Data Analysis (EDA)  
- Data preprocessing & feature engineering  
- Supervised and unsupervised anomaly detection models  
- Explainability (SHAP, feature importance, decision boundaries)  
- Evaluation using precision, recall, F1-score, confusion matrices, ROC curves  
- Reporting and visualization  

The project is structured as a **multi-week, research-oriented workflow**, following the same weekly breakdown used in the course.

---

## Dataset

### NSL-KDD  
The project uses the **NSL-KDD** dataset, an improved version of the original KDD'99 intrusion dataset.  
It removes duplicate records, reduces redundancy, and provides better evaluation consistency.

The dataset includes:

- **41 features** (numeric and categorical)  
- **Labels** indicating whether traffic is *normal* or a specific type of attack  
- **Attack families** such as DoS, Probe, R2L, U2R  

Dataset sources used:

- `KDDTrain+.txt`  
- `KDDTest+.txt`  
- Preprocessed `.npy` artifacts generated throughout the weekly notebooks

---

## Repository Structure

```text
Final_Project/
│
├── data/
│   ├── raw/                # Original NSL-KDD files
│   ├── interim/            # Cleaned CSV/Parquet files
│   └── processed/          # Numpy arrays for models
│
├── notebooks/
|   ├── figures/
│   ├── week_01_eda.ipynb
│   ├── week_02_preprocessing.ipynb
│   ├── week_03_unsupervised.ipynb
│   ├── week_04_models.ipynb
│   ├── week_05_results_and_fixing.ipynb
│   └── week_06_report_explain.ipynb
│
├── src/
│   ├── eval.py
│   ├── explain.py
│   ├── io.py
│   ├── models.py
│   ├── plots.py
│   ├── preps.py
│   ├── unsupervised.py
│   ├── thresholds.py
│   └── utils.py
│
├── reports/
│   ├── week_03
│   ├── week_04
│   ├── week_05
│
├── writeup.ipynb              # write-up report
├── requirements.txt
└── README.md
```

## Weekly Plan of Attack

### **Week 1 — Data Understanding & EDA**
- Load NSL-KDD dataset  
- Inspect feature types, missing values, cardinality  
- Visualize distributions, attack family counts  
- Encode initial labels for binary and multiclass tasks  

### **Week 2 — Preprocessing & Feature Engineering**
- One-hot encoding / label encoding  
- Scaling (MinMax, StandardScaler, RobustScaler)  
- Train/test split  
- Save preprocessed artifacts into `data/processed/`  

### **Week 3 — Unsupervised Models**
- Z-Score anomaly detection  
- Elliptic Envelope  
- Mahalanobis Distance (robust covariance)  
- Local Outlier Factor, Isolation Forest  
- Evaluate unsupervised predictions using confusion matrices  
- Save results for Week 4  

### **Week 4 — Supervised Models**
- Logistic Regression  
- SVM (RBF/Linear)  
- Random Forest, XGBoost  
- Autoencoder (deep learning)  
- Model selection using precision, recall, F1  
- Feature importance & decision boundary visualization  

### **Week 5 — Error Analysis & Fixing**
- Handle NaN propagation from previous weeks  
- Recompute missing cells without running full notebooks  
- Validate saved artifacts  
- Repair pipeline inconsistencies  
- Produce clean evaluation outputs for Week 6  

### **Week 6 — Explainability & Reporting**
- SHAP (Kernel / Tree / DeepExplainer)  
- Global & local feature importance  
- PCA decision boundary plots  
- Compare transparency vs. accuracy  
- Write final project report & conclusions  

---

## Author
**Mehran Tajbakhsh**  
Master of Science in Data Science  
College of Information Science — University of Arizona



#### Disclosure:
Derived from the original data viz course by Mine Çetinkaya-Rundel @ Duke University
