"""
Thresholding utilities for anomaly scores.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve


def percentile_threshold(scores: np.ndarray, q: float) -> float:
    """
    Return the score threshold at a given upper percentile q (0–100).

    Example:
        thr = percentile_threshold(anomaly_scores, 95.0)
    """
    return float(np.percentile(scores, q))


def youden_j_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1,
) -> Tuple[float, float]:
    """
    Compute threshold that maximizes Youden's J statistic on the ROC curve.

    Returns:
        best_threshold, best_J

    This is useful when converting continuous anomaly scores into binary
    anomaly labels in a principled way.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=pos_label)
    J = tpr - fpr
    idx = int(J.argmax())
    return float(thr[idx]), float(J[idx])
