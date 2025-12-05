# %% [markdown]
# # Week 1 — Data Preparation (NSL-KDD)
# This notebook prepares data, encodes features, scales numerics, splits, and saves artifacts.

# %%
# Imports.
from pathlib import Path
import pandas as pd
from sklearn.utils import check_random_state
from src.utils import set_all_seeds, DATA_RAW, DATA_PROCESSED, ensure_dir, RANDOM_STATE
from src.io import load_nsl_kdd_raw, save_numpy, save_joblib
from src.prep import fit_transform_split

# %%
# Set seeds.
set_all_seeds(RANDOM_STATE)

# %%
# Define paths.
raw_file = DATA_RAW / "NSL-KDD.raw"
processed_dir = ensure_dir(DATA_PROCESSED)

# %%
# Load dataset.
df = load_nsl_kdd_raw(raw_file)

# %%
# Quick schema checks.
# Verify column count and presence of categorical columns.
assert df.shape[1] == 42, "Dataset must have 41 features plus label."
for c in ["protocol_type", "service", "flag", "label"]:
    assert c in df.columns, f"Column {c} must exist."

# %%
# Convert numeric columns safely.
# Ensure numeric types for all non-categorical, non-label columns.
categorical = ["protocol_type", "service", "flag"]
numeric = [c for c in df.columns if c not in categorical + ["label"]]
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# %%
# Show basic info.
# Print shapes and simple missing stats.
print("Shape:", df.shape)
print("Missing numeric values:", int(df[numeric].isna().sum().sum()))
print("Label distribution:\n", df["label"].value_counts().head(10))

# %%
# Split and transform.
X_train, X_test, y_train, y_test, arts = fit_transform_split(df, random_state=RANDOM_STATE)

# %%
# Persist arrays.
save_numpy(X_train, processed_dir / "X_train.npy")
save_numpy(X_test, processed_dir / "X_test.npy")
save_numpy(y_train, processed_dir / "y_train.npy")
save_numpy(y_test, processed_dir / "y_test.npy")

# %%
# Persist preprocessor.
save_joblib(arts.preprocessor, processed_dir / "preprocessor.joblib")

# %%
# Final summary.
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train classes:", {k:int(v) for k,v in pd.Series(y_train).value_counts().items()})
print("Test classes:",  {k:int(v) for k,v in pd.Series(y_test).value_counts().items()})
print("Saved to:", processed_dir)
