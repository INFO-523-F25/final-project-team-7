"""
Utility helpers for reproducibility and paths.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import random
import os

# Detect project root as the parent of the src folder.
# This works no matter where the notebook is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42  # Fixed seed as per project standard.

def set_global_seed(seed: int = RANDOM_STATE):
    """Set seeds for numpy and random. (Use in every notebook.)"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass(frozen=True)
class Paths:
    """Centralized project paths based on the project root."""
    root: Path = PROJECT_ROOT
    data_raw: Path = root / "data" / "raw"
    data_proc: Path = root / "data" / "processed"  # matches your folder name
    figs: Path = root / "notebooks" / "figures"
    artifacts: Path = root / "notebooks" / "artifacts"

    def ensure(self):
        """Create directories if missing."""
        for p in [self.data_raw, self.data_proc, self.figs, self.artifacts]:
            p.mkdir(parents=True, exist_ok=True)
        return self
