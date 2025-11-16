#!/usr/bin/env python3
"""
Beat-level SBP/DBP estimation from PPG + ABP using all Part_1 groups.

Usage
-----
python ppg_bp_train_part1.py \
    --data-dir /path/to/split_records \
    --fs 125 \
    --out ./artifacts

What it does
------------
- Scans data-dir for: Part_1_group_*.npz
- For each NPZ:
    - loads all records: record_0, record_1, ...
    - assumes columns: [PPG, ABP, ECG(optional)]
- For each record:
    - bandpass PPG, lowpass ABP
    - detect ABP systolic and diastolic points
    - derive per-beat SBP/DBP labels
    - extract PPG morphology features for each beat:
        ppg_up_amp, ppg_rise_time, ppg_pttd_proxy, ppg_ai,
        ppg_lasi (best-effort), ppg_s1..s4, hr_bpm
- Concatenates all beats across Part_1
- Trains two Gradient Boosting regressors (SBP, DBP) with GroupKFold by record
- Saves models + per-beat predictions + CV metrics.
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# --------------------- Configuration dataclass ---------------------

@dataclass
class Config:
    fs: float = 125.0          # sampling rate (Hz)  <-- adjust if needed
    col_ppg: int = 0           # index of PPG column in record array
    col_abp: int = 1           # index of ABP column
    col_ecg: int | None = 2    # index of ECG column (optional)

    # PPG & ABP filters
    ppg_bp_lo: float = 0.5     # bandpass low cutoff (Hz)
    ppg_bp_hi: float = 8.0     # bandpass high cutoff (Hz)
    abp_lp_hi: float = 20.0    # ABP low-pass cutoff (Hz)

    # Peak detection constraints
    min_hr_bpm: float = 40.0
    max_hr_bpm: float = 180.0
    peak_prominence_abp: float = 10.0   # adjust if peaks are missed
    peak_prominence_ppg: float = 0.1    # not heavily used yet


# ----------------------------- Filters -----------------------------

def butter_bandpass(lo, hi, fs, order=2):
    ny = 0.5 * fs
    b, a = butter(order, [lo/ny, hi/ny], btype="band")
    return b, a

def butter_lowpass(hi, fs, order=2):
    ny = 0.5 * fs
    b, a = butter(order, hi/ny, btype="low")
    return b, a

def apply_filter(sig, b, a):
    if len(sig) < max(3 * len(b), 32):
        return sig
    return filtfilt(b, a, sig)


# --------------------- ABP beats + labels -------------------------

def detect_abp_systolic_peaks(abp, cfg: Config):
    """Detect systolic peaks in ABP."""
    fs = cfg.fs
    min_dist = int(fs * 60.0 / cfg.max_hr_bpm)
    peaks, _ = find_peaks(
        abp,
        distance=max(1, min_dist),
        prominence=cfg.peak_prominence_abp,
    )
    return peaks

def beat_dbp_from_abp(abp, systolic_idx):
    """
    For each systolic peak, define SBP = ABP[peak].
    Define DBP as the minimum ABP between this peak and the next.
    Last beat is dropped if there is no next peak.
    """
    sbp_vals, dbp_vals, sbp_i, dbp_i = [], [], [], []
    for i in range(len(systolic_idx) - 1):
        p = systolic_idx[i]
        q = systolic_idx[i + 1]
        seg = abp[p:q]
        if len(seg) < 3:
            continue
        sbp_vals.append(abp[p])
        sbp_i.append(p)
        rel = np.argmin(seg)
        dbp_vals.append(seg[rel])
        dbp_i.append(p + rel)
    return np.array(sbp_vals), np.array(dbp_vals), np.array(sbp_i), np.array(dbp_i)


# --------------------- PPG morphology helpers ---------------------

def ppg_foot_and_peak(ppg, search_center, fs, win_ms=400):
    """
    Around an ABP systolic index, find PPG foot (onset) and peak.
    Returns (foot_idx, peak_idx) or (None, None).
    """
    half = int(fs * win_ms / 1000.0)
    lo = max(0, search_center - half)
    hi = min(len(ppg), search_center + half)

    seg = ppg[lo:hi]
    if len(seg) < 8:
        return None, None

    # Peak in the right 2/3 of the window
    r_lo = lo + (hi - lo) // 3
    rel_peak = np.argmax(ppg[r_lo:hi])
    peak_i = r_lo + rel_peak

    # Foot: minimum before the peak
    if peak_i <= lo:
        return None, None
    rel_foot = np.argmin(ppg[lo:peak_i])
    foot_i = lo + rel_foot

    if not (lo <= foot_i < peak_i <= hi - 1):
        return None, None
    if ppg[peak_i] <= ppg[foot_i]:
        return None, None

    return foot_i, peak_i

def ppg_max_slope_point(ppg, foot_i, peak_i):
    """Index of maximum rising slope between foot and peak."""
    if foot_i is None or peak_i is None or peak_i - foot_i < 3:
        return None
    seg = np.diff(ppg[foot_i:peak_i+1])
    k = np.argmax(seg)
    return foot_i + k

def ppg_two_peaks_for_AI(ppg, peak_i, fs, max_gap_ms=250):
    """
    Find a second (reflected) peak after systolic for Augmentation Index.
    Returns (y, x) = (diastolic_peak, systolic_peak) or None.
    """
    max_gap = int(fs * max_gap_ms / 1000.0)
    lo = peak_i + 1
    hi = min(len(ppg), peak_i + 1 + max_gap)
    if hi - lo < 8:
        return None
    cand, _ = find_peaks(ppg[lo:hi], prominence=0.02)
    if len(cand) == 0:
        return None
    diast_i = lo + cand[np.argmax(ppg[lo:hi][cand])]
    x = ppg[peak_i]
    y = ppg[diast_i]
    if x <= 0 or y <= 0:
        return None
    return y, x

def ppg_area_segments(ppg, foot_i, peak_i):
    """
    Approximate S1–S4 as areas of four equal bins between foot and peak.
    """
    if foot_i is None or peak_i is None:
        return (np.nan, np.nan, np.nan, np.nan)
    seg = ppg[foot_i:peak_i+1]
    L = len(seg)
    if L < 8:
        return (np.nan, np.nan, np.nan, np.nan)
    bins = np.array_split(seg, 4)
    areas = [float(np.trapz(b)) for b in bins]
    while len(areas) < 4:
        areas.append(np.nan)
    return tuple(areas[:4])

def ppg_feature_dict(ppg, abp, sbp_i, dbp_i, fs, cfg: Config):
    """
    Build per-beat feature dicts aligned to ABP systolic peaks.
    """
    rows = []
    for k in range(len(sbp_i) - 1):
        anchor = sbp_i[k]
        foot_i, peak_i = ppg_foot_and_peak(ppg, anchor, fs)

        # amplitude + rise time
        if foot_i is None or peak_i is None:
            up_amp = np.nan
            rise_t = np.nan
        else:
            up_amp = float(ppg[peak_i] - ppg[foot_i])
            rise_t = float((peak_i - foot_i) / fs)

        # max-slope timing (PTTd proxy)
        maxslope_i = ppg_max_slope_point(ppg, foot_i, peak_i)
        pttd = float((maxslope_i - anchor) / fs) if maxslope_i is not None else np.nan

        # Augmentation Index
        ai_pair = ppg_two_peaks_for_AI(ppg, peak_i if peak_i else anchor, fs)
        ai = float(ai_pair[0] / ai_pair[1]) if ai_pair else np.nan

        # LASI: left as NaN (requires more precise fiducials; optional)
        lasi = np.nan

        # S1–S4
        s1, s2, s3, s4 = ppg_area_segments(ppg, foot_i, peak_i)

        # HR from ABP RR interval
        rr = sbp_i[k+1] - sbp_i[k]
        hr = 60.0 * fs / rr if rr > 0 else np.nan

        rows.append({
            "beat_index": int(k),
            "anchor_sample": int(anchor),
            "ppg_up_amp": up_amp,
            "ppg_rise_time": rise_t,
            "ppg_pttd_proxy": pttd,
            "ppg_ai": ai,
            "ppg_lasi": lasi,
            "ppg_s1": s1,
            "ppg_s2": s2,
            "ppg_s3": s3,
            "ppg_s4": s4,
            "hr_bpm": hr,
            "sbp": float(abp[sbp_i[k]]),
            "dbp": float(abp[dbp_i[k]]),
        })
    return pd.DataFrame(rows)


# -------------------------- Record processing ---------------------

def process_record(arr: np.ndarray, cfg: Config) -> pd.DataFrame:
    """
    arr: (N, C) array -> columns [PPG, ABP, ECG?]
    returns per-beat features+labels DataFrame.
    """
    ppg = arr[:, cfg.col_ppg].astype(float)
    abp = arr[:, cfg.col_abp].astype(float)

    # Filtering
    b_ppg = butter_bandpass(cfg.ppg_bp_lo, cfg.ppg_bp_hi, cfg.fs)
    ppg_f = apply_filter(ppg, *b_ppg)

    b_abp = butter_lowpass(cfg.abp_lp_hi, cfg.fs)
    abp_f = apply_filter(abp, *b_abp)

    # ABP beats
    peaks = detect_abp_systolic_peaks(abp_f, cfg)
    if len(peaks) < 3:
        return pd.DataFrame()

    sbp_vals, dbp_vals, sbp_i, dbp_i = beat_dbp_from_abp(abp_f, peaks)
    if len(sbp_i) < 2 or len(dbp_i) < 2:
        return pd.DataFrame()

    # PPG morphology features
    df = ppg_feature_dict(ppg_f, abp_f, sbp_i, dbp_i, cfg.fs, cfg)
    return df


# ---------------------- Training & evaluation ---------------------

def train_models(df: pd.DataFrame, groups: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    results_dir = out_dir / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    feat_cols = [
        "ppg_up_amp", "ppg_rise_time", "ppg_pttd_proxy",
        "ppg_ai", "ppg_lasi",
        "ppg_s1", "ppg_s2", "ppg_s3", "ppg_s4",
        "hr_bpm",
    ]

    # 1) Drop rows with missing TARGETS only
    dfm = df.dropna(subset=["sbp", "dbp"]).copy()

    X_all = dfm[feat_cols].values
    y_sbp_all = dfm["sbp"].values
    y_dbp_all = dfm["dbp"].values
    grp_all = np.asarray(groups)[:len(dfm)]

    # 2) Drop rows where ALL features are NaN (completely unusable),
    #    but keep rows with partial NaNs (handled by HistGradientBoosting).
    mask = ~np.all(np.isnan(X_all), axis=1)
    X = X_all[mask]
    y_sbp = y_sbp_all[mask]
    y_dbp = y_dbp_all[mask]
    grp = grp_all[mask]
    dfm = dfm.iloc[mask]

    if X.shape[0] < 2:
        raise RuntimeError("Not enough valid beats after filtering for training.")

    # 3) Models that accept NaNs natively
    sbp_model = HistGradientBoostingRegressor(random_state=42)
    dbp_model = HistGradientBoostingRegressor(random_state=42)

    unique_groups = np.unique(grp)
    n_groups = len(unique_groups)

    cv_info = {
        "SBP_pred": [], "DBP_pred": [],
        "SBP_true": [], "DBP_true": [],
        "beat_row": []
    }
    sbp_mae, dbp_mae, sbp_rmse, dbp_rmse = [], [], [], []

    # 4) If we have at least 2 groups, do proper GroupKFold CV.
    if n_groups >= 2:
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)

        for fold, (tr, va) in enumerate(gkf.split(X, y_sbp, groups=grp), 1):
            sbp_model.fit(X[tr], y_sbp[tr])
            dbp_model.fit(X[tr], y_dbp[tr])

            ps = sbp_model.predict(X[va])
            pd_ = dbp_model.predict(X[va])

            sbp_mae.append(mean_absolute_error(y_sbp[va], ps))
            dbp_mae.append(mean_absolute_error(y_dbp[va], pd_))
            sbp_rmse.append(mean_squared_error(y_sbp[va], ps, squared=False))
            dbp_rmse.append(mean_squared_error(y_dbp[va], pd_, squared=False))

            cv_info["SBP_pred"].extend(ps.tolist())
            cv_info["DBP_pred"].extend(pd_.tolist())
            cv_info["SBP_true"].extend(y_sbp[va].tolist())
            cv_info["DBP_true"].extend(y_dbp[va].tolist())
            cv_info["beat_row"].extend(va.tolist())

        metrics = {
            "cv_type": "GroupKFold",
            "sbp_mae_mean": float(np.mean(sbp_mae)),
            "dbp_mae_mean": float(np.mean(dbp_mae)),
            "sbp_rmse_mean": float(np.mean(sbp_rmse)),
            "dbp_rmse_mean": float(np.mean(dbp_rmse)),
            "folds": n_splits,
        }

    else:
        # Only 1 group → no proper CV possible; train once & report train error
        sbp_model.fit(X, y_sbp)
        dbp_model.fit(X, y_dbp)
        ps = sbp_model.predict(X)
        pd_ = dbp_model.predict(X)

        metrics = {
            "cv_type": "train_only",
            "sbp_mae_mean": float(mean_absolute_error(y_sbp, ps)),
            "dbp_mae_mean": float(mean_absolute_error(y_dbp, pd_)),
            "sbp_rmse_mean": float(mean_squared_error(y_sbp, ps, squared=False)),
            "dbp_rmse_mean": float(mean_squared_error(y_dbp, pd_, squared=False)),
            "folds": 1,
        }

        cv_info["SBP_pred"] = ps.tolist()
        cv_info["DBP_pred"] = pd_.tolist()
        cv_info["SBP_true"] = y_sbp.tolist()
        cv_info["DBP_true"] = y_dbp.tolist()
        cv_info["beat_row"] = list(range(len(y_sbp)))

    # 5) Fit final models on ALL cleaned beats
    sbp_model.fit(X, y_sbp)
    dbp_model.fit(X, y_dbp)

    joblib.dump(sbp_model, models_dir / "sbp_model.joblib")
    joblib.dump(dbp_model, models_dir / "dbp_model.joblib")

    with open(models_dir / "feature_meta.json", "w") as f:
        json.dump({"feature_cols": feat_cols}, f, indent=2)
    with open(results_dir / "cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 6) Save per-beat predictions aligned with dfm
    pred_df = dfm.iloc[cv_info["beat_row"]].copy()
    pred_df["SBP_pred"] = cv_info["SBP_pred"]
    pred_df["DBP_pred"] = cv_info["DBP_pred"]
    pred_df["SBP_true"] = cv_info["SBP_true"]
    pred_df["DBP_true"] = cv_info["DBP_true"]
    pred_df.to_csv(results_dir / "beat_predictions.csv", index=False)

    return metrics




# ---------------------------- Main logic -------------------------

def load_all_part1(data_dir: Path):
    """
    Scan data_dir for Part_1_group_*.npz and load all records.
    Returns list of (record_array, record_global_id).
    """
    npz_files = sorted(data_dir.glob("Part_1_group_*.npz"))
    records = []
    for g_idx, npz_path in enumerate(npz_files):
        z = np.load(npz_path, allow_pickle=True)
        # keys like record_0, record_1, ...
        rec_keys = [k for k in z.files if k.startswith("record_")]
        rec_keys.sort(key=lambda x: int(x.split("_")[1]))
        for r_k in rec_keys:
            arr = z[r_k]
            global_id = f"group{g_idx}_rec{r_k.split('_')[1]}"
            records.append((arr, global_id))
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing Part_1_group_*.npz")
    parser.add_argument("--fs", type=float, default=125.0,
                        help="Sampling frequency (Hz)")
    parser.add_argument("--out", type=Path, default=Path("./artifacts"),
                        help="Output directory for models/results")
    args = parser.parse_args()

    cfg = Config(fs=args.fs)

    # 1) Load all records from Part_1 groups
    recs = load_all_part1(args.data_dir)
    if not recs:
        raise RuntimeError(f"No Part_1_group_*.npz files found in {args.data_dir}")

    all_rows = []
    all_groups = []

    # 2) Process each record into beat-level features
    for idx, (arr, rec_id) in enumerate(recs):
        df_rec = process_record(arr, cfg)
        if df_rec.empty:
            continue
        df_rec["record_id"] = rec_id
        df_rec["global_record_idx"] = idx
        all_rows.append(df_rec)
        all_groups.extend([idx] * len(df_rec))

    if not all_rows:
        raise RuntimeError("No usable beats found. Try adjusting filter/peak params.")

    big = pd.concat(all_rows, ignore_index=True)

    # 3) Train models & save artifacts
    metrics = train_models(big, np.array(all_groups), args.out)

    print("Training complete.")
    print("Artifacts saved under:", args.out)
    print("CV metrics:", metrics)


if __name__ == "__main__":
    main()
