"""
Evaluation utilities for Week 04 ML models.

Includes:
- Confusion matrix plotting
- Classification report wrapper
- ROC + Precision-Recall curves
- Score → anomaly label converter
"""

from __future__ import annotations
from typing import Sequence, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Confusion matrix plotter
# ---------------------------------------------------------------------------

def plot_confusion(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[Sequence] = None,
    title: str = "Confusion matrix",
    ax: Optional[plt.Axes] = None,
    normalize: bool = False,
):
    """
    Plot confusion matrix with optional normalization.
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="YlGnBu",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)

    return ax


# ---------------------------------------------------------------------------
# Classification report wrapper
# ---------------------------------------------------------------------------

def classification_summary(
    y_true: Sequence,
    y_pred: Sequence,
    target_names: Optional[Sequence[str]] = None,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Print classification report and also return as a dict.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    if print_report:
        print(classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
        ))
    return report


# ---------------------------------------------------------------------------
# ROC & Precision–Recall curves (binary)
# ---------------------------------------------------------------------------

def binary_roc_pr_curves(
    y_true: Sequence[int],
    y_score: Sequence[float],
    pos_label: int = 1,
    title_prefix: str = "",
):
    """
    Plot ROC curve and PR curve for anomaly scores or classifier probabilities.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], ls="--", alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{title_prefix}ROC (AUC={auc:.3f})")

    # PR
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix}Precision-Recall")

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Convert anomaly scores → binary labels
# ---------------------------------------------------------------------------

def anomaly_labels_from_scores(
    scores: Sequence[float],
    threshold: float,
    anomaly_is_high: bool = True,
) -> np.ndarray:
    """
    Convert continuous anomaly scores to binary labels.

    Args:
        scores: list/array of anomaly scores
        threshold: cutoff
        anomaly_is_high: if True, scores >= threshold → anomaly (1)

    """
    scores = np.asarray(scores)
    if anomaly_is_high:
        labels = (scores >= threshold).astype(int)
    else:
        labels = (scores <= threshold).astype(int)
    return labels
