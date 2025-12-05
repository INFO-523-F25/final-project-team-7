
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Dict, Any

from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def compute_rf_feature_importance(model, feature_names):
    """Return Random Forest Gini importances as a DataFrame.

    Parameters
    ----------
    model : fitted RandomForestClassifier or similar
        Must expose feature_importances_.
    feature_names : list of str
        Names of features in the same order as the model input.

    Returns
    -------
    pd.DataFrame with columns [feature, importance], sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no feature_importances_ attribute.")
    importances = np.asarray(model.feature_importances_)
    if len(feature_names) != len(importances):
        feature_names = [f"f{i}" for i in range(len(importances))]
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def compute_permutation_importance(
    model,
    X,
    y,
    feature_names,
    n_repeats: int = 10,
    max_samples: Optional[int] = 3000,
    random_state: int = 42,
    scoring: str = "f1_macro",
):
    """Compute permutation importance on a (possibly subsampled) dataset.

    Parameters
    ----------
    model : fitted estimator
    X, y : arrays
    feature_names : list of str
    n_repeats : int
        Number of permutation repetitions.
    max_samples : int or None
        If not None, subsample up to this many rows for speed.
    scoring : str
        Scoring metric for permutation_importance.

    Returns
    -------
    pd.DataFrame with columns [feature, importance], sorted descending.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    if max_samples is not None and n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub, y_sub = X, y

    result = permutation_importance(
        model,
        X_sub,
        y_sub,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    importances = result.importances_mean
    if len(feature_names) != len(importances):
        feature_names = [f"f{i}" for i in range(len(importances))]
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def compute_kernel_shap(
    model,
    X_background,
    X_explain,
    feature_names,
    max_background: int = 200,
    max_explain: int = 200,
) -> Dict[str, Any]:
    """Compute Kernel SHAP for a black-box model (e.g., SVM).

    Returns a dict containing:
    - "shap_values": SHAP values array
    - "explainer": the KernelExplainer object
    - "summary_fig": matplotlib Figure for the summary plot
    - "local_plots": list of matplotlib Figures for a few local explanations
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is not installed. Install with `pip install shap`.") from e

    X_background = np.asarray(X_background)
    X_explain = np.asarray(X_explain)

    # Subsample for speed
    if X_background.shape[0] > max_background:
        X_background = X_background[:max_background]
    if X_explain.shape[0] > max_explain:
        X_explain = X_explain[:max_explain]

    # For multi-class, SHAP expects predict_proba
    if hasattr(model, "predict_proba"):
        f = model.predict_proba
    else:
        # Fallback to decision_function; Kernel SHAP will treat it as output
        f = model.decision_function if hasattr(model, "decision_function") else model.predict

    explainer = shap.KernelExplainer(f, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples="auto")

    # Summary plot (assume multi-class => list of arrays; take average magnitude)
    shap_values_arr = shap_values
    if isinstance(shap_values, list):
        # Aggregate across classes by L1 norm
        shap_values_arr = np.mean([np.abs(v) for v in shap_values], axis=0)

    fig_summary = plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values_arr,
        X_explain,
        feature_names=feature_names,
        show=False
    )

    # A few local plots (waterfall) for the first examples
    local_plots = []
    max_local = min(3, X_explain.shape[0])
    for i in range(max_local):
        fig = plt.figure(figsize=(6, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_arr[i],
                base_values=np.mean(explainer.expected_value)
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value,
                data=X_explain[i],
                feature_names=feature_names,
            ),
            show=False,
        )
        local_plots.append(fig)

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "summary_fig": fig_summary,
        "local_plots": local_plots,
    }


def compute_deep_shap(
    model,
    X_background,
    X_explain,
    feature_names,
    max_background: int = 200,
    max_explain: int = 200,
) -> Dict[str, Any]:
    """Compute Deep SHAP for a deep autoencoder-like model.

    Assumes a Keras-like model with a .predict method.
    Returns the same dict structure as compute_kernel_shap.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is not installed. Install with `pip install shap`.") from e

    X_background = np.asarray(X_background)
    X_explain = np.asarray(X_explain)

    if X_background.shape[0] > max_background:
        X_background = X_background[:max_background]
    if X_explain.shape[0] > max_explain:
        X_explain = X_explain[:max_explain]

    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)

    shap_values_arr = shap_values
    if isinstance(shap_values, list):
        shap_values_arr = np.mean([np.abs(v) for v in shap_values], axis=0)

    fig_summary = plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values_arr,
        X_explain,
        feature_names=feature_names,
        show=False
    )

    local_plots = []
    max_local = min(3, X_explain.shape[0])
    for i in range(max_local):
        fig = plt.figure(figsize=(6, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_arr[i],
                base_values=np.mean(explainer.expected_value)
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value,
                data=X_explain[i],
                feature_names=feature_names,
            ),
            show=False,
        )
        local_plots.append(fig)

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "summary_fig": fig_summary,
        "local_plots": local_plots,
    }


def plot_anomaly_score_distributions(
    wk3_unsup: Optional[pd.DataFrame] = None,
    wk4_unsup: Optional[pd.DataFrame] = None,
    max_models: int = 6,
):
    """Plot anomaly score distributions for several unsupervised models.

    Parameters
    ----------
    wk3_unsup, wk4_unsup : DataFrames (may be None)
        Each must contain columns ['model', 'score'].
    max_models : int
        Maximum number of distinct models to plot.

    Returns
    -------
    fig : matplotlib Figure
    """
    frames = []
    for df in [wk3_unsup, wk4_unsup]:
        if df is not None:
            frames.append(df[["model", "score"]].copy())

    if not frames:
        raise ValueError("No unsupervised DataFrames provided.")

    all_df = pd.concat(frames, ignore_index=True)
    # Choose up to max_models distinct model names
    models = all_df["model"].value_counts().index[:max_models]

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        sub = all_df[all_df["model"] == m]["score"]
        sns.kdeplot(sub, ax=ax, label=m, fill=False)

    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly score distributions (selected models)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_reconstruction_error_hist(scores, bins: int = 50):
    """Plot histogram of Autoencoder reconstruction errors.

    Parameters
    ----------
    scores : array-like
        Reconstruction error per sample.
    bins : int
        Number of histogram bins.

    Returns
    -------
    fig : matplotlib Figure
    """
    scores = np.asarray(scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=bins, edgecolor="black")
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder reconstruction error distribution")
    plt.tight_layout()
    return fig


def compute_pca_embedding(X, n_components: int = 2, random_state: int = 42):
    """Return a 2D PCA embedding of X (for visualization only)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca


def plot_pca_dbscan_like(
    X_pca: np.ndarray,
    labels_pred,
    labels_true,
    title: str,
    save_path=None,
):
    """Scatter plot of PCA embedding colored by predicted anomaly label.

    Parameters
    ----------
    X_pca : array (n_samples, 2)
        2D PCA coordinates of X_test.
    labels_pred : array-like
        Predicted binary labels (0=normal, 1=anomaly) from a DBSCAN-like model.
    labels_true : array-like
        True binary labels for reference.
    """
    X_pca = np.asarray(X_pca)
    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels_pred,
        cmap="coolwarm",
        alpha=0.6,
        s=10,
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted (0=normal, 1=anomaly)")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pca_decision_boundaries(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    random_state: int = 42,
):
    """Plot decision boundaries of RF and SVM in 2D PCA space.

    This function:
    1) Computes PCA(2) on X_train.
    2) Projects both train and test into PCA space.
    3) Fits *lightweight proxy* RF and SVM on the PCA coordinates.
    4) Plots decision regions and test points colored by true label.

    Returns
    -------
    fig : matplotlib Figure
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    pca = PCA(n_components=2, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Lightweight proxy models for visualization
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        random_state=random_state,
    )

    rf.fit(X_train_pca, y_train)
    svm.fit(X_train_pca, y_train)

    # Define grid
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    rf_pred = rf.predict(grid).reshape(xx.shape)
    svm_pred = svm.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    models = [("RF (PCA proxy)", rf_pred), ("SVM-RBF (PCA proxy)", svm_pred)]

    for ax, (title, Z) in zip(axes, models):
        # Decision region
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        # Test points colored by true class
        scatter = ax.scatter(
            X_test_pca[:, 0],
            X_test_pca[:, 1],
            c=y_test,
            cmap="tab10",
            edgecolor="k",
            s=15,
            alpha=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

    plt.tight_layout()
    return fig
