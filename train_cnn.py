# import os
# import glob
# import warnings
# warnings.filterwarnings("ignore")  # suppress warnings

# import numpy as np
# from scipy.signal import butter, filtfilt, find_peaks
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from tqdm.auto import tqdm   # progress bars

# # ====================== CONFIG ======================

# SPLIT_DIR = r"split_records/"  # folder with Part_1_group_*.npz
# FS = 125.0

# # Heart-rate bounds
# MIN_BPM = 40
# MAX_BPM = 180

# # Windowing
# WINDOW_SEC = 8.0    # length in seconds of each training window
# HOP_SEC    = 4.0    # hop between windows
# MIN_BEATS_PER_WIN = 3  # require at least this many beats per window

# BATCH_SIZE = 64
# EPOCHS = 50
# LR = 1e-3
# VAL_SPLIT = 0.2
# RANDOM_SEED = 42

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ==================== SIGNAL HELPERS ====================

# def butter_bandpass(lo, hi, fs, order=2):
#     ny = 0.5 * fs
#     b, a = butter(order, [lo/ny, hi/ny], btype="band")
#     return b, a

# def butter_lowpass(hi, fs, order=2):
#     ny = 0.5 * fs
#     b, a = butter(order, hi/ny, btype="low")
#     return b, a

# def apply_filter(x, b, a):
#     if len(x) < max(3 * len(b), 32):
#         return x
#     return filtfilt(b, a, x)

# def find_ppg_troughs(ppg, fs):
#     """Detect PPG troughs (valleys) via -PPG peaks."""
#     min_distance = int(fs * 60.0 / MAX_BPM)
#     trough_idx, _ = find_peaks(-ppg, distance=min_distance, prominence=0.05)
#     return trough_idx

# # ==================== LABELING HELPERS ====================

# def compute_beat_sbp_dbp(ppg, abp, trough_idx, fs):
#     """
#     Given full PPG/ABP and trough indices, compute per-beat SBP/DBP.
#     Returns lists: beats = [(start, end)], sbp_list, dbp_list
#     """
#     shortest_len = int(fs * 60.0 / MAX_BPM)
#     longest_len = int(fs * 60.0 / MIN_BPM)

#     beats = []
#     sbp_list = []
#     dbp_list = []

#     for k in range(len(trough_idx) - 1):
#         start = trough_idx[k]
#         end   = trough_idx[k + 1]
#         if end <= start:
#             continue

#         beat_len = end - start + 1
#         if beat_len < shortest_len or beat_len > longest_len:
#             continue

#         abp_seg = abp[start:end + 1]
#         sbp = float(np.max(abp_seg))
#         dbp = float(np.min(abp_seg))

#         beats.append((start, end))
#         sbp_list.append(sbp)
#         dbp_list.append(dbp)

#     return beats, np.array(sbp_list, dtype=np.float32), np.array(dbp_list, dtype=np.float32)

# def extract_windows_from_record(ppg_raw, abp_raw, fs,
#                                 window_sec, hop_sec, min_beats_per_win):
#     """
#     From one record, return list of (ppg_window, [SBP_win, DBP_win]) pairs.
#     - filter signals
#     - detect beats on full record
#     - slide window over time; within each window, aggregate beat SBP/DBP
#     """
#     # Filter signals
#     b_ppg, a_ppg = butter_bandpass(0.5, 8.0, fs, order=2)
#     ppg = apply_filter(ppg_raw, b_ppg, a_ppg)

#     b_abp, a_abp = butter_lowpass(20.0, fs, order=2)
#     abp = apply_filter(abp_raw, b_abp, a_abp)

#     # Detect troughs & beats
#     trough_idx = find_ppg_troughs(ppg, fs)
#     if len(trough_idx) < 2:
#         return []

#     beats, sbp_beats, dbp_beats = compute_beat_sbp_dbp(ppg, abp, trough_idx, fs)
#     if len(beats) == 0:
#         return []

#     win_len = int(window_sec * fs)
#     hop_len = int(hop_sec * fs)
#     N = len(ppg)

#     samples = []

#     for start in range(0, N - win_len + 1, hop_len):
#         end = start + win_len  # end is exclusive

#         # find beats fully inside this window
#         beat_indices = []
#         for i, (b_start, b_end) in enumerate(beats):
#             if b_start >= start and b_end < end:
#                 beat_indices.append(i)

#         if len(beat_indices) < min_beats_per_win:
#             continue

#         sbp_win = sbp_beats[beat_indices]
#         dbp_win = dbp_beats[beat_indices]

#         # aggregate with median for robustness
#         sbp_label = float(np.median(sbp_win))
#         dbp_label = float(np.median(dbp_win))

#         ppg_seg = ppg[start:end].astype(np.float32)

#         # per-window z-normalization (you can experiment with per-record instead)
#         m = ppg_seg.mean()
#         s = ppg_seg.std()
#         if s < 1e-6:
#             continue
#         ppg_norm = (ppg_seg - m) / s

#         samples.append((ppg_norm, [sbp_label, dbp_label]))

#     return samples

# # ==================== DATASET BUILD ====================

# def build_dataset_from_split(split_dir, fs, window_sec, hop_sec, min_beats_per_win):
#     X = []
#     y = []   # [SBP, DBP]
#     file_paths = sorted(glob.glob(os.path.join(split_dir, "Part_1_group_*.npz")))
#     if len(file_paths) == 0:
#         raise RuntimeError("No NPZ files found in split_dir.")

#     for npz_path in file_paths:
#         data = np.load(npz_path, allow_pickle=True)
#         for key in data.files:
#             rec = data[key]

#             # Only use 2D arrays with at least 2 columns: [PPG, ABP, ...]
#             if not isinstance(rec, np.ndarray):
#                 continue
#             if rec.ndim != 2 or rec.shape[1] < 2:
#                 continue

#             ppg_raw = rec[:, 0].astype(float)
#             abp_raw = rec[:, 1].astype(float)

#             samples = extract_windows_from_record(
#                 ppg_raw, abp_raw, fs,
#                 window_sec=window_sec,
#                 hop_sec=hop_sec,
#                 min_beats_per_win=min_beats_per_win
#             )

#             for ppg_win, bp_pair in samples:
#                 X.append(ppg_win)
#                 y.append(bp_pair)

#     if len(X) == 0:
#         raise RuntimeError("No valid windows found. Check data and thresholds.")

#     X = np.stack(X, axis=0)              # (num_windows, T)
#     y = np.array(y, dtype=np.float32)    # (num_windows, 2)
#     print(f"Built dataset: {X.shape[0]} windows, sequence length = {X.shape[1]}")
#     return X, y

# class PPGWindowDataset(Dataset):
#     """Window-level PPG dataset for joint SBP+DBP prediction."""
#     def __init__(self, X, y, y_mean=None, y_std=None):
#         # X: (N, T), y: (N, 2)
#         self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, T)

#         if y_mean is None or y_std is None:
#             self.y_mean = torch.from_numpy(y.mean(axis=0).astype(np.float32))
#             self.y_std  = torch.from_numpy(y.std(axis=0).astype(np.float32) + 1e-6)
#         else:
#             self.y_mean = y_mean
#             self.y_std  = y_std

#         y_norm = (y - self.y_mean.numpy()) / self.y_std.numpy()
#         self.y = torch.from_numpy(y_norm.astype(np.float32))  # (N, 2)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # ==================== CNN MODEL ====================

# class CNNRegressor(nn.Module):
#     """
#     1D CNN: input = PPG window (multi-beat), output = [SBP, DBP] (standardized).
#     """
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),

#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),

#             nn.Conv1d(64, 128, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),  # -> (B, 128, 1)
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),              # (B, 128)
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 2)           # SBP, DBP (standardized)
#         )

#     def forward(self, x):
#         # x: (B, 1, T)
#         h = self.conv(x)
#         out = self.fc(h)
#         return out
    
# class CNNBiLSTMRegressor(nn.Module):
#     """
#     CNN + BiLSTM: input = PPG window (multi-beat), output = [SBP, DBP] (standardized).
#     - CNN: local feature extraction over time
#     - BiLSTM: temporal modeling over CNN feature sequence
#     """

#     def __init__(self,
#                  num_conv_channels=64,
#                  lstm_hidden=64,
#                  lstm_layers=1,
#                  bidirectional=True,
#                  dropout=0.3):
#         super().__init__()

#         # 1D CNN feature extractor (keeps time resolution)
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),          # T -> T/2

#             nn.Conv1d(32, num_conv_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),          # T/2 -> T/4

#             # one more conv block if you want a bit more abstraction
#             nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#         )

#         self.bidirectional = bidirectional
#         num_dirs = 2 if bidirectional else 1

#         # BiLSTM over the CNN feature sequence
#         self.lstm = nn.LSTM(
#             input_size=num_conv_channels,
#             hidden_size=lstm_hidden,
#             num_layers=lstm_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#             dropout=dropout if lstm_layers > 1 else 0.0,
#         )

#         # FC head from last timestep
#         self.fc = nn.Sequential(
#             nn.Linear(lstm_hidden * num_dirs, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 2)   # standardized [SBP, DBP]
#         )

#     def forward(self, x):
#         # x: (B, 1, T_raw)
#         h = self.conv(x)              # (B, C, T_cnn)
#         h = h.transpose(1, 2)         # (B, T_cnn, C) for LSTM

#         lstm_out, _ = self.lstm(h)    # (B, T_cnn, H*num_dirs)
#         last = lstm_out[:, -1, :]     # take last timestep representation

#         out = self.fc(last)           # (B, 2)
#         return out


# # ==================== TRAIN / EVAL ====================

# def train_model(X, y):
#     np.random.seed(RANDOM_SEED)
#     torch.manual_seed(RANDOM_SEED)

#     # Split by windows (for best generalization, prefer splitting by subject/file)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED
#     )

#     # Fit normalization on train targets
#     y_mean = y_train.mean(axis=0).astype(np.float32)
#     y_std = (y_train.std(axis=0) + 1e-6).astype(np.float32)

#     y_mean_t = torch.from_numpy(y_mean)
#     y_std_t  = torch.from_numpy(y_std)

#     train_ds = PPGWindowDataset(X_train, y_train, y_mean=y_mean_t, y_std=y_std_t)
#     val_ds   = PPGWindowDataset(X_val,   y_val,   y_mean=y_mean_t, y_std=y_std_t)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

#     # model = CNNRegressor().to(DEVICE)
#     model = CNNBiLSTMRegressor().to(DEVICE)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, factor=0.5, patience=5
#     )

#     print(f"\n===== Training CNN SBP+DBP model on {DEVICE} =====")
#     print(model)

#     best_val_rmse = float("inf")
#     best_state = None

#     for epoch in range(1, EPOCHS + 1):
#         # ---- train ----
#         model.train()
#         train_loss = 0.0
#         train_bar = tqdm(train_loader,
#                          desc=f"Epoch {epoch}/{EPOCHS} (train)",
#                          leave=False)
#         for xb, yb in train_bar:
#             xb = xb.to(DEVICE)
#             yb = yb.to(DEVICE)  # standardized targets

#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * xb.size(0)

#         train_loss /= len(train_ds)

#         # ---- validate ----
#         model.eval()
#         val_loss = 0.0
#         all_true = []
#         all_pred = []

#         val_bar = tqdm(val_loader,
#                        desc=f"Epoch {epoch}/{EPOCHS} (val)  ",
#                        leave=False)
#         with torch.no_grad():
#             for xb, yb in val_bar:
#                 xb = xb.to(DEVICE)
#                 yb = yb.to(DEVICE)

#                 preds = model(xb)
#                 loss = criterion(preds, yb)
#                 val_loss += loss.item() * xb.size(0)

#                 # de-standardize for metrics
#                 preds_real = preds.cpu().numpy() * y_std + y_mean
#                 y_real = yb.cpu().numpy() * y_std + y_mean

#                 all_true.append(y_real)
#                 all_pred.append(preds_real)

#         val_loss /= len(val_ds)
#         all_true = np.concatenate(all_true, axis=0)
#         all_pred = np.concatenate(all_pred, axis=0)

#         # metrics for SBP (col 0) and DBP (col 1)
#         mae_sbp = mean_absolute_error(all_true[:, 0], all_pred[:, 0])
#         mae_dbp = mean_absolute_error(all_true[:, 1], all_pred[:, 1])
#         rmse_sbp = np.sqrt(mean_squared_error(all_true[:, 0], all_pred[:, 0]))
#         rmse_dbp = np.sqrt(mean_squared_error(all_true[:, 1], all_pred[:, 1]))

#         # simple aggregate RMSE for scheduler / checkpoint
#         agg_rmse = 0.5 * (rmse_sbp + rmse_dbp)
#         scheduler.step(agg_rmse)

#         print(
#             f"Epoch {epoch:02d} | Train MSE={train_loss:.3f} | "
#             f"Val MSE={val_loss:.3f} | "
#             f"MAE(SBP)={mae_sbp:.2f}, MAE(DBP)={mae_dbp:.2f} | "
#             f"RMSE(SBP)={rmse_sbp:.2f}, RMSE(DBP)={rmse_dbp:.2f}"
#         )

#         if agg_rmse < best_val_rmse:
#             best_val_rmse = agg_rmse
#             best_state = {
#                 "model": model.state_dict(),
#                 "y_mean": y_mean,
#                 "y_std": y_std,
#             }

#     # Save best model
#     os.makedirs("deep_bp_artifacts", exist_ok=True)
#     model_path = os.path.join("deep_bp_artifacts", "ppg_cnn_sbp_dbp.pt")
#     torch.save(best_state, model_path)
#     print(f"Saved best model to {model_path} (best avg RMSE = {best_val_rmse:.2f})")

#     return model

# def main():
#     print("Building window-level dataset from split records...")
#     X, y = build_dataset_from_split(
#         SPLIT_DIR, FS,
#         window_sec=WINDOW_SEC,
#         hop_sec=HOP_SEC,
#         min_beats_per_win=MIN_BEATS_PER_WIN
#     )
#     _ = train_model(X, y)

# if __name__ == "__main__":
#     main()


import os
import glob
import warnings
warnings.filterwarnings("ignore")  # suppress warnings

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm   # progress bars

# ====================== CONFIG ======================

SPLIT_DIR = r"split_records/"  # folder with Part_*_group_*.npz
FS = 125.0

# Heart-rate bounds
MIN_BPM = 40
MAX_BPM = 180

# Windowing
WINDOW_SEC = 8.0    # length in seconds of each training window
HOP_SEC    = 4.0    # hop between windows
MIN_BEATS_PER_WIN = 3  # require at least this many beats per window

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== SIGNAL HELPERS ====================

def butter_bandpass(lo, hi, fs, order=2):
    ny = 0.5 * fs
    b, a = butter(order, [lo/ny, hi/ny], btype="band")
    return b, a

def butter_lowpass(hi, fs, order=2):
    ny = 0.5 * fs
    b, a = butter(order, hi/ny, btype="low")
    return b, a

def apply_filter(x, b, a):
    if len(x) < max(3 * len(b), 32):
        return x
    return filtfilt(b, a, x)

def find_ppg_troughs(ppg, fs):
    """Detect PPG troughs (valleys) via -PPG peaks."""
    min_distance = int(fs * 60.0 / MAX_BPM)
    trough_idx, _ = find_peaks(-ppg, distance=min_distance, prominence=0.05)
    return trough_idx

# ==================== LABELING HELPERS ====================

def compute_beat_sbp_dbp(ppg, abp, trough_idx, fs):
    """
    Given full PPG/ABP and trough indices, compute per-beat SBP/DBP.
    Returns lists: beats = [(start, end)], sbp_list, dbp_list
    """
    shortest_len = int(fs * 60.0 / MAX_BPM)
    longest_len = int(fs * 60.0 / MIN_BPM)

    beats = []
    sbp_list = []
    dbp_list = []

    for k in range(len(trough_idx) - 1):
        start = trough_idx[k]
        end   = trough_idx[k + 1]
        if end <= start:
            continue

        beat_len = end - start + 1
        if beat_len < shortest_len or beat_len > longest_len:
            continue

        abp_seg = abp[start:end + 1]
        sbp = float(np.max(abp_seg))
        dbp = float(np.min(abp_seg))

        beats.append((start, end))
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    return beats, np.array(sbp_list, dtype=np.float32), np.array(dbp_list, dtype=np.float32)

def extract_windows_from_record(ppg_raw, abp_raw, fs,
                                window_sec, hop_sec, min_beats_per_win):
    """
    From one record, return list of (ppg_window, [SBP_win, DBP_win]) pairs.
    - filter signals
    - detect beats on full record
    - slide window over time; within each window, aggregate beat SBP/DBP
    """
    # Filter signals
    b_ppg, a_ppg = butter_bandpass(0.5, 8.0, fs, order=2)
    ppg = apply_filter(ppg_raw, b_ppg, a_ppg)

    b_abp, a_abp = butter_lowpass(20.0, fs, order=2)
    abp = apply_filter(abp_raw, b_abp, a_abp)

    # Detect troughs & beats
    trough_idx = find_ppg_troughs(ppg, fs)
    if len(trough_idx) < 2:
        return []

    beats, sbp_beats, dbp_beats = compute_beat_sbp_dbp(ppg, abp, trough_idx, fs)
    if len(beats) == 0:
        return []

    win_len = int(window_sec * fs)
    hop_len = int(hop_sec * fs)
    N = len(ppg)

    samples = []

    for start in range(0, N - win_len + 1, hop_len):
        end = start + win_len  # end is exclusive

        # find beats fully inside this window
        beat_indices = []
        for i, (b_start, b_end) in enumerate(beats):
            if b_start >= start and b_end < end:
                beat_indices.append(i)

        if len(beat_indices) < min_beats_per_win:
            continue

        sbp_win = sbp_beats[beat_indices]
        dbp_win = dbp_beats[beat_indices]

        # aggregate with median for robustness
        sbp_label = float(np.median(sbp_win))
        dbp_label = float(np.median(dbp_win))

        ppg_seg = ppg[start:end].astype(np.float32)

        # per-window z-normalization
        m = ppg_seg.mean()
        s = ppg_seg.std()
        if s < 1e-6:
            continue
        ppg_norm = (ppg_seg - m) / s

        samples.append((ppg_norm, [sbp_label, dbp_label]))

    return samples

# ==================== DATASET BUILD ====================

def build_dataset_from_split(split_dir, fs, window_sec, hop_sec, min_beats_per_win):
    X = []
    y = []   # [SBP, DBP]

    # Use both Part 1 and Part 2 npz files
    pattern = os.path.join(split_dir, "Part_*_group_*.npz")
    file_paths = sorted(glob.glob(pattern))
    if len(file_paths) == 0:
        raise RuntimeError(f"No NPZ files found in {split_dir} with pattern {pattern!r}.")

    print("Using NPZ files:")
    for p in file_paths:
        print("  ", os.path.basename(p))

    for npz_path in file_paths:
        data = np.load(npz_path, allow_pickle=True)
        for key in data.files:
            rec = data[key]

            # Only use 2D arrays with at least 2 columns: [PPG, ABP, ...]
            if not isinstance(rec, np.ndarray):
                continue
            if rec.ndim != 2 or rec.shape[1] < 2:
                continue

            ppg_raw = rec[:, 0].astype(float)
            abp_raw = rec[:, 1].astype(float)

            samples = extract_windows_from_record(
                ppg_raw, abp_raw, fs,
                window_sec=window_sec,
                hop_sec=hop_sec,
                min_beats_per_win=min_beats_per_win
            )

            for ppg_win, bp_pair in samples:
                X.append(ppg_win)
                y.append(bp_pair)

    if len(X) == 0:
        raise RuntimeError("No valid windows found. Check data and thresholds.")

    X = np.stack(X, axis=0)              # (num_windows, T)
    y = np.array(y, dtype=np.float32)    # (num_windows, 2)
    print(f"Built dataset: {X.shape[0]} windows, sequence length = {X.shape[1]}")
    return X, y

class PPGWindowDataset(Dataset):
    """Window-level PPG dataset for joint SBP+DBP prediction."""
    def __init__(self, X, y, y_mean=None, y_std=None):
        # X: (N, T), y: (N, 2)
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, T)

        if y_mean is None or y_std is None:
            self.y_mean = torch.from_numpy(y.mean(axis=0).astype(np.float32))
            self.y_std  = torch.from_numpy(y.std(axis=0).astype(np.float32) + 1e-6)
        else:
            self.y_mean = y_mean
            self.y_std  = y_std

        y_norm = (y - self.y_mean.numpy()) / self.y_std.numpy()
        self.y = torch.from_numpy(y_norm.astype(np.float32))  # (N, 2)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== CNN + BiLSTM MODEL ====================

class CNNBiLSTMRegressor(nn.Module):
    """
    CNN + BiLSTM: input = PPG window (multi-beat), output = [SBP, DBP] (standardized).
    - CNN: local feature extraction over time
    - BiLSTM: temporal modeling over CNN feature sequence
    """

    def __init__(self,
                 num_conv_channels=64,
                 lstm_hidden=64,
                 lstm_layers=1,
                 bidirectional=True,
                 dropout=0.3):
        super().__init__()

        # 1D CNN feature extractor (keeps time resolution up to pooling)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # T -> T/2

            nn.Conv1d(32, num_conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # T/2 -> T/4

            nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1

        # BiLSTM over the CNN feature sequence
        self.lstm = nn.LSTM(
            input_size=num_conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # FC head from last timestep
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * num_dirs, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)   # standardized [SBP, DBP]
        )

    def forward(self, x):
        # x: (B, 1, T_raw)
        h = self.conv(x)              # (B, C, T_cnn)
        h = h.transpose(1, 2)         # (B, T_cnn, C) for LSTM

        lstm_out, _ = self.lstm(h)    # (B, T_cnn, H*num_dirs)
        last = lstm_out[:, -1, :]     # take last timestep representation

        out = self.fc(last)           # (B, 2)
        return out
    
class AttentionBlock(nn.Module):
    """
    Additive attention over sequence: produces a weighted sum of LSTM outputs.
    input:  (B, T, H)
    output: (B, H)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        # h: (B, T, H)
        score = self.v(torch.tanh(self.W(h)))   # (B, T, 1)
        weights = torch.softmax(score, dim=1)   # importance weights
        context = (weights * h).sum(dim=1)      # weighted sum → (B, H)
        return context, weights


class CNNBiLSTMAttentionRegressor(nn.Module):
    """
    CNN + BiLSTM + Additive Attention → regression head.
    Output = standardized [SBP, DBP].
    """
    def __init__(self,
                 num_conv_channels=64,
                 lstm_hidden=64,
                 lstm_layers=1,
                 bidirectional=True,
                 dropout=0.3):
        super().__init__()

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, num_conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1
        lstm_input_size = num_conv_channels
        lstm_output_size = lstm_hidden * num_dirs

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Attention
        self.attention = AttentionBlock(lstm_output_size)

        # Final regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (B, 1, T_raw)
        h = self.conv(x)          # (B, C, T_cnn)
        h = h.transpose(1, 2)     # (B, T_cnn, C)

        lstm_out, _ = self.lstm(h)   # (B, T_cnn, H_out)
        context, attn_weights = self.attention(lstm_out)  # (B, H_out)

        out = self.fc(context)
        return out

# ==================== TRAIN / EVAL ====================

def train_model(X, y):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Split by windows (for best generalization, prefer splitting by subject/file)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    # Fit normalization on train targets
    y_mean = y_train.mean(axis=0).astype(np.float32)
    y_std = (y_train.std(axis=0) + 1e-6).astype(np.float32)

    y_mean_t = torch.from_numpy(y_mean)
    y_std_t  = torch.from_numpy(y_std)

    train_ds = PPGWindowDataset(X_train, y_train, y_mean=y_mean_t, y_std=y_std_t)
    val_ds   = PPGWindowDataset(X_val,   y_val,   y_mean=y_mean_t, y_std=y_std_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # model = CNNBiLSTMRegressor().to(DEVICE)
    model = CNNBiLSTMAttentionRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    print(f"\n===== Training CNN+BiLSTM SBP+DBP model on {DEVICE} =====")
    print(model)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch}/{EPOCHS} (train)",
                         leave=False)
        for xb, yb in train_bar:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)  # standardized targets

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        all_true = []
        all_pred = []

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch}/{EPOCHS} (val)  ",
                       leave=False)
        with torch.no_grad():
            for xb, yb in val_bar:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

                # de-standardize for metrics
                preds_real = preds.cpu().numpy() * y_std + y_mean
                y_real = yb.cpu().numpy() * y_std + y_mean

                all_true.append(y_real)
                all_pred.append(preds_real)

        val_loss /= len(val_ds)
        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)

        # metrics for SBP (col 0) and DBP (col 1)
        mae_sbp = mean_absolute_error(all_true[:, 0], all_pred[:, 0])
        mae_dbp = mean_absolute_error(all_true[:, 1], all_pred[:, 1])
        rmse_sbp = np.sqrt(mean_squared_error(all_true[:, 0], all_pred[:, 0]))
        rmse_dbp = np.sqrt(mean_squared_error(all_true[:, 1], all_pred[:, 1]))

        # simple aggregate RMSE for scheduler / checkpoint
        agg_rmse = 0.5 * (rmse_sbp + rmse_dbp)
        scheduler.step(agg_rmse)

        print(
            f"Epoch {epoch:02d} | Train MSE={train_loss:.3f} | "
            f"Val MSE={val_loss:.3f} | "
            f"MAE(SBP)={mae_sbp:.2f}, MAE(DBP)={mae_dbp:.2f} | "
            f"RMSE(SBP)={rmse_sbp:.2f}, RMSE(DBP)={rmse_dbp:.2f}"
        )

        if agg_rmse < best_val_rmse:
            best_val_rmse = agg_rmse
            best_state = {
                "model": model.state_dict(),
                "y_mean": y_mean,
                "y_std": y_std,
            }

    # Save best model
    os.makedirs("deep_bp_artifacts", exist_ok=True)
    model_path = os.path.join("deep_bp_artifacts", "ppg_cnn_bilstm_sbp_dbp.pt")
    torch.save(best_state, model_path)
    print(f"Saved best model to {model_path} (best avg RMSE = {best_val_rmse:.2f})")

    return model

def main():
    print("Building window-level dataset from split records (Part 1 + Part 2)...")
    X, y = build_dataset_from_split(
        SPLIT_DIR, FS,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        min_beats_per_win=MIN_BEATS_PER_WIN
    )
    _ = train_model(X, y)

if __name__ == "__main__":
    main()
