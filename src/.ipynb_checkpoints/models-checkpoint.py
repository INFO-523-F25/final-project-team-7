"""
Model builders and parameter grids for Week 04 ML models.

This module provides:
- Isolation Forest tuning grid
- One-Class SVM tuning grid (with StandardScaler)
- Random Forest tuning grid (class-weight balanced)
- SVM (RBF) tuning grid (with StandardScaler, class-weight balanced)
- Deep Autoencoder builder for anomaly detection

These functions are *additive*: you can copy them into an existing file
without overwriting unrelated code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------
# Isolation Forest (Week04)
# --------------------------------------------------------
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest

@dataclass
class IsoSpec:
    model: IsolationForest
    param_grid: dict

def make_isolation_forest_grid():
    """
    IsolationForest with a small, fast grid for Week04.
    """
    model = IsolationForest(
        behaviour="deprecated",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_samples": ["auto", 0.5],
        "contamination": [0.05, 0.1],
    }

    return IsoSpec(model=model, param_grid=param_grid)

# ---------------------------------------------------------------------------
# Shared configuration structure
# ---------------------------------------------------------------------------

@dataclass
class ModelWithGrid:
    model: Any
    param_grid: Dict[str, Any]


# ---------------------------------------------------------------------------
# Unsupervised / semi-supervised models
# ---------------------------------------------------------------------------

def make_isolation_forest_grid(random_state: int = 42) -> ModelWithGrid:
    """
    Build IsolationForest + parameter grid.

    Returns:
        ModelWithGrid(model, param_grid)
    """
    model = IsolationForest(
        random_state=random_state,
        n_estimators=200,
        contamination=0.05,
    )
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_samples": [0.5, 1.0],
        "contamination": [0.01, 0.05, 0.1],
    }
    return ModelWithGrid(model=model, param_grid=param_grid)


def make_oneclass_svm_grid() -> ModelWithGrid:
    """
    One-Class SVM in a scaling pipeline (StandardScaler → OCSVM).

    Returns:
        ModelWithGrid(model_pipeline, param_grid)
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(kernel="rbf")),
        ]
    )
    param_grid = {
        "ocsvm__nu": [0.05, 0.1, 0.2],
        "ocsvm__gamma": ["scale", 1e-3, 1e-2],
    }
    return ModelWithGrid(model=pipe, param_grid=param_grid)

def make_fast_oneclass_svm() -> Pipeline:
    """
    Fast OC-SVM pipeline for when you don't want to run GridSearchCV.

    This is what you can use for the "option 4" approach in Week04:

        from src.models import make_fast_oneclass_svm

        ocsvm = make_fast_oneclass_svm()

        # Train only on normal data
        mask_normal = (y_train_bin == 0)
        X_train_normal = X_train[mask_normal]

        ocsvm.fit(X_train_normal)
        pred = ocsvm.predict(X_test)
        y_pred = (pred == -1).astype(int)

    Returns:
        Pipeline(StandardScaler → OneClassSVM) with reasonable defaults.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(
                kernel="rbf",
                nu=0.1,        # reasonable default
                gamma="scale"  # reasonable default
            )),
        ]
    )
    return pipe
# ---------------------------------------------------------------------------
# Supervised models (multi-class classification)
# ---------------------------------------------------------------------------

def make_rf_classifier_grid(random_state: int = 42) -> ModelWithGrid:
    """
    Random Forest (class-weight balanced) + grid.

    Returns:
        ModelWithGrid
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 20, 40],
    }
    return ModelWithGrid(model=rf, param_grid=param_grid)


def make_svm_rbf_grid() -> ModelWithGrid:
    """
    SVM RBF classifier wrapped in StandardScaler pipeline.

    Returns:
        ModelWithGrid
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", probability=False)),
        ]
    )
    param_grid = {
        "svc__C": [1, 10, 50],
        "svc__gamma": ["scale", 1e-3, 1e-2],
    }
    return ModelWithGrid(model=pipe, param_grid=param_grid)

def make_fast_svm_rbf():
    """
    Fast SVM-RBF baseline: no GridSearch, good default hyperparameters.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                class_weight="balanced",
                C=10,
                gamma="scale"
            )),
        ]
    )
    return pipe

# ---------------------------------------------------------------------------
# Deep autoencoder (Keras)
# ---------------------------------------------------------------------------

# Keras import is optional so notebook doesn't break if TensorFlow isn't installed
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    keras = None
    layers = None


def build_deep_autoencoder(input_dim: int, latent_dim: int = 16):
    """
    Build a dense autoencoder with symmetric encoder/decoder.

    Returns:
        autoencoder, encoder, decoder
    """
    if keras is None or layers is None:
        raise ImportError("TensorFlow/Keras is required for the autoencoder.")

    # Encoder
    input_layer = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dense(32, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(input_layer, output_layer, name="deep_autoencoder")
    encoder = keras.Model(input_layer, latent, name="encoder")

    # Separate decoder model
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return autoencoder, encoder, decoder
