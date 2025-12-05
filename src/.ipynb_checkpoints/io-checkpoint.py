# src/io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from .utils import ensure_dir

# Canonical NSL-KDD column names (41 features + label).
NSL_KDD_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]

def load_nsl_kdd_raw(path: str | Path) -> pd.DataFrame:
    """Load NSL-KDD from CSV-like raw file with or without header."""
    path = Path(path)
    # Try with header first.
    try:
        df = pd.read_csv(path)
        if "label" not in df.columns or df.shape[1] != 42:
            raise ValueError("Header mismatch. Falling back to no header.")
    except Exception:
        df = pd.read_csv(path, header=None, names=NSL_KDD_COLS)
    return df

def save_joblib(obj, path: str | Path) -> Path:
    """Persist object with joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path

def load_joblib(path: str | Path):
    """Load object persisted with joblib."""
    return joblib.load(path)

# src/io.py
from pathlib import Path
import os
import numpy as np
from .utils import ensure_dir

def save_numpy(arr, path, allow_pickle=False):
    """
    Robustly save NumPy array on Windows:
    - Ensures parent dir exists
    - Saves sparse matrices via .npz
    - Handles object arrays safely
    - Uses explicit 'wb' file handle + atomic rename
    """
    path = Path(path)
    ensure_dir(path.parent)

    # 0) Guard: filename shouldn't be a reserved Windows name
    bad_names = {"CON","PRN","AUX","NUL","COM1","LPT1","COM2","LPT2","COM3","LPT3","COM4","LPT4","COM5","LPT5","COM6","LPT6","COM7","LPT7","COM8","LPT8","COM9","LPT9"}
    if path.stem.upper() in bad_names:
        raise OSError(f"Invalid file name on Windows: {path.name}")

    # 1) Sparse -> save_npz
    try:
        import scipy.sparse as sp
        if sp.issparse(arr):
            from scipy.sparse import save_npz
            npz_path = path.with_suffix(".npz")
            save_npz(str(npz_path), arr)
            return npz_path
    except Exception:
        pass

    # 2) Prepare array
    a = np.asarray(arr)
    if a.dtype == object:
        # unwrap singleton; otherwise allow pickle
        if a.size == 1:
            a = a.item()
            a = np.asarray(a)
        else:
            allow_pickle = True
    # numeric arrays: compact & contiguous
    if np.issubdtype(np.asarray(a).dtype, np.number):
        a = np.ascontiguousarray(a)

    # 3) Atomic write via temp file
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(str(tmp_path), "wb") as f:       # explicit file handle
            np.save(f, a, allow_pickle=allow_pickle)
        # Replace/rename atomically
        if path.exists():
            os.replace(str(tmp_path), str(path))
        else:
            tmp_path.rename(path)
    finally:
        # Clean up temp file if it still exists
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return path
