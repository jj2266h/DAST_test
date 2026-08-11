"""
Shared utilities for CMAPSS FD004 analysis.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Patch threadpoolctl before sklearn is imported to suppress Windows DLL error
try:
    import threadpoolctl as _tpc
    # Wrap info() so any OSError returns an empty list instead of crashing
    _orig_info = _tpc.ThreadpoolController.info
    def _safe_info(self):
        result = []
        for ctrl in getattr(self, 'lib_controllers', []):
            try:
                result.append(ctrl.info())
            except OSError:
                pass
        return result
    _tpc.ThreadpoolController.info = _safe_info
except Exception:
    pass

import json
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans as _sp_kmeans, vq as _sp_vq, whiten as _sp_whiten

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\Users\hjs92\Desktop\DAST_test\Cmapss_data"
ANALYSIS_DIR = r"C:\Users\hjs92\Desktop\DAST_test\analysis"
FIG_DIR      = os.path.join(ANALYSIS_DIR, "figures")
STATS_FILE   = os.path.join(ANALYSIS_DIR, "stats.json")

os.makedirs(FIG_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
COLS         = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS  = [f's{i}' for i in range(1, 22)]
OP_COLS      = ['op1', 'op2', 'op3']
RUL_CAP      = 125
N_CONDITIONS = 6
WINDOW       = 30           # rolling-window size for feature engineering

# Sensors known to carry no degradation information (literature + domain knowledge)
FORCED_EXCLUDE = {'s1', 's5', 's6', 's10', 's16', 's18', 's19'}

# ── Stats I/O ─────────────────────────────────────────────────────────────────
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_stats(stats: dict):
    existing = load_stats()
    existing.update(stats)
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(
        os.path.join(DATA_DIR, "train_FD004.txt"),
        sep=r'\s+', header=None, names=COLS
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "test_FD004.txt"),
        sep=r'\s+', header=None, names=COLS
    )
    rul_df = pd.read_csv(
        os.path.join(DATA_DIR, "RUL_FD004.txt"),
        sep=r'\s+', header=None, names=['RUL_true']
    )
    rul_df['unit'] = rul_df.index + 1
    return train, test, rul_df

# ── RUL Label (piecewise linear, capped) ──────────────────────────────────────
def add_rul(df, cap=RUL_CAP):
    max_cycle = df.groupby('unit')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycle, on='unit')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=cap)
    df.drop(columns='max_cycle', inplace=True)
    return df

# ── Operating Condition Clustering (scipy-based, avoids sklearn BLAS crash) ───
class _OcModel:
    """Thin wrapper around scipy K-Means result for reuse."""
    def __init__(self, centers_whitened, scales):
        self.centers_whitened = centers_whitened   # (k, 3)
        self.scales = scales                        # std of each column
        # Also store un-whitened centres for display
        self.cluster_centers_ = centers_whitened * scales

def fit_oc_kmeans(train_df):
    X = train_df[OP_COLS].values.astype(float)
    scales = X.std(axis=0).clip(1e-9)
    X_w = X / scales
    np.random.seed(42)
    centers_w, _ = _sp_kmeans(X_w, N_CONDITIONS, iter=50, seed=42)
    # Sort by op1 mean for reproducible labelling
    order = centers_w[:, 0].argsort()
    centers_w = centers_w[order]
    km = _OcModel(centers_w, scales)
    return km

def assign_oc(df, km):
    df = df.copy()
    X = df[OP_COLS].values.astype(float) / km.scales
    labels, _ = _sp_vq(X, km.centers_whitened)
    df['OC'] = labels
    return df

# ── Sensor Validity ────────────────────────────────────────────────────────────
def get_valid_sensors(train_df, std_threshold=0.01):
    stds = train_df[SENSOR_COLS].std()
    # Exclude near-constant sensors AND domain-known uninformative sensors
    invalid = [s for s in SENSOR_COLS
               if stds[s] <= std_threshold or s in FORCED_EXCLUDE]
    valid   = [s for s in SENSOR_COLS if s not in invalid]
    return valid, invalid

# ── OC-based Normalisation ────────────────────────────────────────────────────
def compute_oc_stats(train_df, valid_sensors):
    stats = {}
    for oc in sorted(train_df['OC'].unique()):
        subset = train_df[train_df['OC'] == oc][valid_sensors]
        std    = subset.std().replace(0, 1)
        stats[int(oc)] = {'mean': subset.mean().to_dict(),
                          'std':  std.to_dict()}
    return stats

def normalize_by_oc(df, oc_stats, valid_sensors):
    df = df.copy()
    for s in valid_sensors:
        df[f'{s}_norm'] = np.nan
    for oc, stat in oc_stats.items():
        mask = df['OC'] == int(oc)
        for s in valid_sensors:
            df.loc[mask, f'{s}_norm'] = (
                (df.loc[mask, s] - stat['mean'][s]) / stat['std'][s]
            )
    return df

# ── Rolling Features (per engine) ────────────────────────────────────────────
def add_rolling_features(df, norm_cols, window=WINDOW):
    """Add rolling mean and std per engine for normalised sensor columns."""
    result = []
    for _, grp in df.groupby('unit', sort=False):
        grp = grp.sort_values('cycle').copy()
        for col in norm_cols:
            grp[f'{col}_rmean'] = grp[col].rolling(window, min_periods=1).mean()
            grp[f'{col}_rstd']  = grp[col].rolling(window, min_periods=1).std().fillna(0)
        result.append(grp)
    return pd.concat(result).reset_index(drop=True)
