
"""
Plot helpers for EDA (matplotlib / seaborn).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def dist_plots(df: pd.DataFrame, cols, out_dir: Path):
    """Draw histogram + KDE for selected numeric columns."""
    for c in cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {c}")
        ax.set_xlabel(c)
        save_fig(fig, out_dir / f"eda_dist_{c}.png")

def corr_heatmap(df: pd.DataFrame, cols, out_path: Path, method:str="pearson"):
    """Correlation heatmap for selected columns."""
    corr = df[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
    ax.set_title(f"{method.title()} Correlation Heatmap")
    save_fig(fig, out_path)

def topn_bar(series: pd.Series, n:int, out_path: Path, title:str):
    """Top-N frequency bar for a categorical series."""
    vc = series.value_counts().head(n)
    fig, ax = plt.subplots()
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count")
    save_fig(fig, out_path)
