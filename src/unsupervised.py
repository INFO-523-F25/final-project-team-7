"""
Classical anomaly detection helpers (Z-score, Elliptic Envelope, LOF, DBSCAN)
designed to mirror Chapter_2.ipynb behavior on the NSL-KDD-style dataframe.
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd

from sklearn.covariance import EllipticEnvelope, MinCovDet, EmpiricalCovariance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

# Attack family used in the Chapter_2-style binary experiments
BINARY_ATTACK = "teardrop"


def make_binary_subset(
    df: pd.DataFrame,
    normal_label: str = "normal",
    attack_label: str = BINARY_ATTACK,
) -> pd.DataFrame:
    """
    Create a binary subset containing only normal vs a single attack type.
    Adds column 'Label' with 0 for normal, 1 for attack, matching Chapter_2 logic.

    Expects cleaned label column named 'label' (lowercase, no dot).
    """
    sub = df.loc[df["label"].isin([normal_label, attack_label])].copy()

    def map_label(lbl: str) -> int:
        return 0 if lbl == normal_label else 1

    sub["Label"] = sub["label"].apply(map_label)
    return sub


def z_score_feature(series: pd.Series) -> pd.Series:
    """Compute Z-score for a single feature."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0:
        sigma = 1.0
    return (series - mu) / sigma


def z_score_labels(z: pd.Series, k: float = 2.0) -> np.ndarray:
    """
    Map Z-scores to binary labels as in Chapter_2:
    1 if z > k or z < -k, else 0.
    """
    return np.where((z > k) | (z < -k), 1, 0)


@dataclass
class EllipticResult:
    confusion: np.ndarray
    predictions: np.ndarray


def run_elliptic_envelope(
    df_bin: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 0,
) -> EllipticResult:
    """
    Mirror Chapter_2 EllipticEnvelope:
    - Use all numeric features except Label/label and categorical protocol/service/flag/family.
    - Return confusion matrix and rescored predictions (1 = outlier, 0 = inlier).
    """
    actual = df_bin["Label"].to_numpy()

    drop_cols = [
        c
        for c in [
            "Label",
            "label",
            "target",
            "protocol_type",
            "service",
            "flag",
            "family",
        ]
        if c in df_bin.columns
    ]
    X = df_bin.drop(columns=drop_cols)

    clf = EllipticEnvelope(contamination=contamination, random_state=random_state)
    pred = clf.fit_predict(X)

    # Chapter_2 convention: -1 -> outlier (1), +1 -> inlier (0)
    pred_rescored = np.where(pred == -1, 1, 0)

    cm = confusion_matrix(actual, pred_rescored)
    return EllipticResult(confusion=cm, predictions=pred_rescored)


@dataclass
class LOFGridResult:
    ks: List[int]
    accuracies: List[float]
    precisions: List[float]
    recalls: List[float]


def lof_grid_search(
    df_bin: pd.DataFrame,
    k_values: List[int],
) -> LOFGridResult:
    """
    Grid search over n_neighbors for LocalOutlierFactor, mirroring Chapter_2 loop.

    Metrics use the same simple formulas as the notebook:
    - Accuracy  = (TP + TN) / N
    - Precision = TP / (TP + FP + 1)
    - Recall    = TP / (TP + FN + 1)
    (The +1 avoids division by zero exactly as in many teaching examples.)
    """
    actual = df_bin["Label"].to_numpy()

    drop_cols = [
        c
        for c in [
            "Label",
            "label",
            "target",
            "protocol_type",
            "service",
            "flag",
            "family",
        ]
        if c in df_bin.columns
    ]
    X = df_bin.drop(columns=drop_cols)
    total = len(X)

    all_k, all_acc, all_prec, all_rec = [], [], [], []

    for k in k_values:
        clf = LocalOutlierFactor(n_neighbors=k, contamination=0.1)
        pred = clf.fit_predict(X)
        pred_rescored = np.where(pred == -1, 1, 0)

        cm = confusion_matrix(actual, pred_rescored)
        tn, fp, fn, tp = cm.ravel()

        acc = 100.0 * (tp + tn) / total
        prec = 100.0 * tp / (tp + fp + 1)
        rec = 100.0 * tp / (tp + fn + 1)

        all_k.append(k)
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)

    return LOFGridResult(
        ks=all_k,
        accuracies=all_acc,
        precisions=all_prec,
        recalls=all_rec,
    )


@dataclass
class DBSCANResult:
    confusion: np.ndarray
    predictions: np.ndarray

# ----------------------------------------------------------------------
# Mahalanobis distance (robust covariance + chi-square threshold)
# ----------------------------------------------------------------------

@dataclass
class MahalanobisResult:
    distances_sq: np.ndarray
    threshold: float
    confusion: np.ndarray
    predictions: np.ndarray


def run_mahalanobis(
    df_bin: pd.DataFrame,
    alpha: float = 0.99,
    robust: bool = True,
) -> MahalanobisResult:
    """
    Mahalanobis distance-based anomaly detection.

    - Fits either a robust covariance (MinCovDet) or classical EmpiricalCovariance.
    - Uses squared Mahalanobis distances.
    - Threshold is chi-square quantile with dof = n_features at level 'alpha'.
    """
    from scipy.stats import chi2
    actual = df_bin["Label"].to_numpy()

    drop_cols = [
        c
        for c in [
            "Label",
            "label",
            "target",
            "protocol_type",
            "service",
            "flag",
            "family",
        ]
        if c in df_bin.columns
    ]

    X = df_bin.drop(columns=drop_cols).to_numpy()
    n_features = X.shape[1]

    # Robust (Minimum Covariance Determinant) or classical covariance
    if robust:
        cov = MinCovDet().fit(X)
    else:
        cov = EmpiricalCovariance().fit(X)

    distances_sq = cov.mahalanobis(X)

    threshold = chi2.ppf(alpha, df=n_features)

    pred = (distances_sq > threshold).astype(int)

    cm = confusion_matrix(actual, pred)

    return MahalanobisResult(
        distances_sq=distances_sq,
        threshold=threshold,
        confusion=cm,
        predictions=pred,
    )



def run_dbscan(
    df_bin: pd.DataFrame,
    eps: float = 0.2,
    min_samples: int = 5,
) -> DBSCANResult:
    """
    Run DBSCAN-based anomaly detection similar to Chapter_2:
    - Uses the same feature set as LOF/Elliptic (drops Label and categoricals).
    - Rescores labels so noise points (cluster label == -1) are anomalies (1).
    """
    actual = df_bin["Label"].to_numpy()

    drop_cols = [
        c
        for c in [
            "Label",
            "label",
            "target",
            "protocol_type",
            "service",
            "flag",
            "family",
        ]
        if c in df_bin.columns
    ]
    X = df_bin.drop(columns=drop_cols)

    clf = DBSCAN(eps=eps, min_samples=min_samples)
    pred = clf.fit_predict(X)

    pred_rescored = np.where(pred == -1, 1, 0)
    cm = confusion_matrix(actual, pred_rescored)

    return DBSCANResult(confusion=cm, predictions=pred_rescored)
