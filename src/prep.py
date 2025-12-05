
"""
Preprocessing: encode categoricals and scale numerics.
"""
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

CATEGORICAL = ["protocol_type","service","flag"]

@dataclass
class PrepArtifacts:
    ohe_path: str
    scaler_path: str
    x_train_path: str
    x_test_path: str
    y_train_path: str
    y_test_path: str

def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Create a ColumnTransformer that one-hot encodes categoricals and passes numerics."""
    numeric_cols = [c for c in df.columns if c not in CATEGORICAL + ["label","family"]]
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    # Pipeline ensures StandardScaler applies after OHE to numerics only
    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, CATEGORICAL),
            ("num", StandardScaler(with_mean=True, with_std=True), numeric_cols),
        ],
        remainder="drop"
    )
    return pre, numeric_cols

def split_and_fit(df: pd.DataFrame, paths, random_state:int=42, stratify_on:str="family") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """Stratified 80/20 split on family, then fit transformer on train only and transform both."""
    X = df.drop(columns=["label","family"])
    y = df[stratify_on]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pre, numeric_cols = build_preprocessor(df)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    # Persist encoder & scaler embedded in ColumnTransformer
    joblib.dump(pre, paths.data_proc / "preprocessor.joblib")

    return X_train_t, X_test_t, y_train.to_numpy(), y_test.to_numpy(), pre
