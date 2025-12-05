# Jupyter Notebooks (`notebooks/`)

This folder contains all **Jupyter notebooks** used for the analysis, experimentation, and modeling stages of the project  
**“Anomaly Detection in Network Traffic Using the NSL-KDD Dataset.”**

Week-01 objective:
Prepare the NSL-KDD network intrusion dataset for later analysis and modeling by performing consistent preprocessing and storing reproducible training and testing artifacts.
Task Completed:
| Step                        | Description                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **1. Load Data**            | Loaded `NSL-KDD.raw` from `data/raw/` using canonical 41 feature + label format.                            |
| **2. Schema Validation**    | Verified column count = 42 and existence of critical columns (`protocol_type`, `service`, `flag`, `label`). |
| **3. Type Conversion**      | Converted non-categorical fields to numeric; coerced invalid values to NaN.                                 |
| **4. Label Mapping**        | Created high-level `family` label (`normal`, `dos`, `probe`, `r2l`, `u2r`, `other`).                        |
| **5. Train-Test Split**     | Stratified 80/20 split on `family` with `RANDOM_STATE = 42`.                                                |
| **6. Encoding + Scaling**   | One-hot encoded categorical features and standardized numeric features via `ColumnTransformer`.             |
| **7. Artifact Persistence** | Saved preprocessed arrays and preprocessor object for reuse.                                                |
Week-02 objective:

Perform exploratory data analysis (EDA) on the NSL-KDD dataset to understand its structure, composition, and key feature relationships. Generate tabular summaries, visualize feature distributions and correlations, and apply dimensionality-reduction techniques to reveal data separability among attack families.

| Step                             | Description                                                                                                                                                                                   |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Environment Setup**         | Added a bootstrap cell to dynamically locate the project root and make the `src` package importable from the notebook.                                                                        |
| **2. Data Import**               | Imported `load_nsl_kdd_raw` from `src/io.py`, and used constants and utilities (`DATA_RAW`, `FIGURES`, `set_all_seeds`, `ensure_dir`) from `src/utils.py`.                                    |
| **3. Load Dataset**              | Loaded the NSL-KDD dataset from `data/raw/NSL-KDD.raw` and verified the presence of all 42 columns (`41 features + 1 label`).                                                                 |
| **4. Data Preview**              | Displayed the top five rows of the dataset using `df.head()` for quick inspection of structure and value patterns.                                                                            |
| **5. Label Distribution**        | Displayed the distribution of samples by attack type (`label`) sorted descending to assess dataset imbalance.                                                                                 |
| **6. Numeric Conversion**        | Converted non-categorical fields to numeric types with coercion to handle invalid values, ensuring numeric consistency.                                                                       |
| **7. Family Mapping**            | Derived a new categorical column `family` using `to_family()` from `src/prep.py`, grouping fine-grained attacks into broader categories: `normal`, `dos`, `probe`, `r2l`, `u2r`, and `other`. |
| **8. Descriptive Exploration**   | Computed missing-value counts, displayed total missing numeric entries, and printed counts per attack family.                                                                                 |
| **9. Numeric Distributions**     | Created histograms for key numeric features such as `duration`, `src_bytes`, `dst_bytes`, `count`, and others to visualize spread and skewness.                                               |
| **10. Correlation Analysis**     | Computed the correlation matrix among numeric features and visualized it as a heatmap to detect strong dependencies.                                                                          |
| **11. Categorical Frequencies**  | Plotted frequency distributions for categorical features (`protocol_type`, `service`, `flag`) and family imbalance for class visualization.                                                   |
| **12. Preprocessing Pipeline**   | Built a `ColumnTransformer` via `build_preprocessor()` from `src/prep.py` to one-hot encode categorical features and scale numeric ones.                                                      |
| **13. Stratified Sampling**      | Implemented a custom stratified sampling approach to draw up to 20,000 representative records without using deprecated `groupby.apply`.                                                       |
| **14. Dimensionality Reduction** | Applied **PCA**, **t-SNE**, and **UMAP** (if available) to visualize clusters and separability across families in 2D projections.                                                             |
| **15. Visualization Saving**     | Saved all figures automatically to `notebooks/figures/` using descriptive filenames (e.g., `eda_pca_2d.png`, `eda_cat_flag.png`, etc.).                                                       |
| **16. Reproducibility**          | Maintained full reproducibility using `RANDOM_STATE = 42` and modular functions from `src/` for clean, reusable workflow.                                                                     |

Figures:
- eda_dist_*.png — histograms of key numeric features
- eda_corr_matrix.png — numeric correlation heatmap
- eda_cat_protocol_type.png, eda_cat_flag.png, eda_cat_service_top.png — categorical frequencies
- eda_family_distribution.png — class imbalance visualization
- eda_pca_2d.png, eda_tsne_2d.png, eda_umap_2d.png — dimensionality reduction results

