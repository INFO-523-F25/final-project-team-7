
"""
Data I/O utilities for NSL-KDD corrected 10% file.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import joblib

# Official KDD'99 feature names (41) + label.
FEATURES_41 = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]
LABEL_COL = "label"

FAMILY_MAP: Dict[str, str] = {
    # Normal
    "normal": "normal",
    # DoS
    "neptune": "DoS","smurf": "DoS","back": "DoS","teardrop": "DoS","pod": "DoS","land": "DoS",
    "apache2": "DoS","processtable": "DoS","udpstorm": "DoS","worm":"DoS",
    # Probe
    "satan": "Probe","ipsweep": "Probe","nmap": "Probe","portsweep": "Probe","mscan":"Probe","saint":"Probe",
    # R2L
    "guess_passwd": "R2L","ftp_write": "R2L","imap":"R2L","phf":"R2L","multihop":"R2L",
    "warezmaster":"R2L","warezclient":"R2L","spy":"R2L","xlock":"R2L","xsnoop":"R2L","snmpguess":"R2L",
    # U2R
    "buffer_overflow":"U2R","loadmodule":"U2R","rootkit":"U2R","perl":"U2R","httptunnel":"U2R","ps":"U2R",
    "sqlattack":"U2R","xterm":"U2R"
}

def load_raw_nsl_kdd(path: Path) -> pd.DataFrame:
    """Load NSL-KDD 'corrected' 10% file from data\\raw. Handle comma/space delim and headerless files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None, sep=r"[,\s]+", engine="python")
    # If file already has header, respect it; else set names.
    if df.shape[1] == 42:
        df.columns = FEATURES_41 + [LABEL_COL]
    elif LABEL_COL in df.columns and len(df.columns) == 42:
        pass
    else:
        # Try to coerce last column as label
        if df.shape[1] >= 42:
            df = df.iloc[:, :42]
            df.columns = FEATURES_41 + [LABEL_COL]
        else:
            raise ValueError("Unexpected column count; expected 42 including label.")
    # Clean label (strip periods etc.)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower().str.replace(".", "", regex=False)
    return df

def map_attack_family(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'family' column based on attack label mapping."""
    fam = df[LABEL_COL].map(lambda x: FAMILY_MAP.get(x, "unknown"))
    df = df.copy()
    df["family"] = fam
    return df

def save_numpy(path: Path, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def save_joblib(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
