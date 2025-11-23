import os
import glob
import warnings
warnings.filterwarnings("ignore")  # 1) suppress warnings

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm   # 2) tqdm for progress bars

# ====================== CONFIG ======================

SPLIT_DIR = r"E:\Masters_College_Work\Semester_4\Embedded_Deep_Learning\18844_Embedded_Machine_Learning-Project\split_records"
FS = 125.0

# Heart-rate bounds → valid beat lengths
MIN_BPM = 40
MAX_BPM = 180

# Max number of samples we allow for a single PPG cycle
MAX_CYCLE_LEN = int(FS * 60.0 / MIN_BPM)   # longest expected beat

BATCH_SIZE = 64
EPOCHS = 30
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

def extract_cycles_from_record(ppg_raw, abp_raw, fs, max_cycle_len):
    """
    From one record, return list of (ppg_cycle_padded, sbp, dbp).
    - PPG/ABP filtered
    - cycles defined trough→trough on PPG
    - SBP = max(ABP) in cycle, DBP = min(ABP) in cycle
    - PPG cycle z-normalized then zero-padded to max_cycle_len
    """
    # Filter signals
    b_ppg, a_ppg = butter_bandpass(0.5, 8.0, fs, order=2)
    ppg = apply_filter(ppg_raw, b_ppg, a_ppg)

    b_abp, a_abp = butter_lowpass(20.0, fs, order=2)
    abp = apply_filter(abp_raw, b_abp, a_abp)

    trough_idx = find_ppg_troughs(ppg, fs)
    if len(trough_idx) < 2:
        return []

    samples = []
    shortest_len = int(fs * 60.0 / MAX_BPM)  # shortest allowed beat
    longest_len  = max_cycle_len            # longest allowed beat

    for k in range(len(trough_idx) - 1):
        start = trough_idx[k]
        end   = trough_idx[k + 1]
        if end <= start:
            continue

        ppg_seg = ppg[start:end + 1]
        abp_seg = abp[start:end + 1]

        beat_len = len(ppg_seg)
        if beat_len < shortest_len or beat_len > longest_len:
            continue  # skip noisy/odd beats

        # SBP / DBP in this cycle
        sbp = float(np.max(abp_seg))
        dbp = float(np.min(abp_seg))

        # per-beat z-normalization *before* padding
        m = ppg_seg.mean()
        s = ppg_seg.std()
        if s < 1e-6:
            continue
        ppg_norm = (ppg_seg - m) / s

        # 3) zero-pad to max_cycle_len (no time rescaling)
        ppg_padded = np.zeros(max_cycle_len, dtype=np.float32)
        ppg_padded[:beat_len] = ppg_norm.astype(np.float32)

        samples.append((ppg_padded, sbp, dbp))

    return samples

# ==================== DATASET BUILD ====================

def build_dataset_from_split(split_dir, fs, max_cycle_len):
    X = []
    y = []   # [SBP, DBP]
    file_paths = sorted(glob.glob(os.path.join(split_dir, "Part_1_group_*.npz")))

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

            samples = extract_cycles_from_record(ppg_raw, abp_raw, fs, max_cycle_len)
            for ppg_cycle, sbp, dbp in samples:
                X.append(ppg_cycle)
                y.append([sbp, dbp])

    if len(X) == 0:
        raise RuntimeError("No valid cycles found. Check data and thresholds.")

    X = np.stack(X, axis=0)              # (num_cycles, T)
    y = np.array(y, dtype=np.float32)    # (num_cycles, 2)
    print(f"Built dataset: {X.shape[0]} cycles, sequence length = {X.shape[1]}")
    return X, y

class PPGDatasetScalar(Dataset):
    """Per-cycle PPG dataset for a single scalar target (SBP or DBP)."""
    def __init__(self, X, y_scalar):
        # X: (N, T), y_scalar: (N,)
        self.X = torch.from_numpy(X).unsqueeze(1)                 # (N, 1, T)
        self.y = torch.from_numpy(y_scalar.astype(np.float32))    # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== LSTM MODEL ====================

class LSTMRegressor(nn.Module):
    """
    3) LSTM model: input = 1D PPG cycle (with zero padding),
       output = scalar (SBP OR DBP).
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 bidirectional=True, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        num_dirs = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_dirs, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, 1, T) -> (B, T, 1) for LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # use last timestep representation
        last = out[:, -1, :]          # (B, hidden*num_dirs)
        y = self.fc(last).squeeze(1)  # (B,)
        return y

# ==================== TRAIN / EVAL ====================

def train_single_target(X, y_scalar, target_name):
    """
    4) Train a separate LSTM model for SBP or DBP.
    y_scalar is 1D array: SBP or DBP.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_scalar, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    train_ds = PPGDatasetScalar(X_train, y_train)
    val_ds   = PPGDatasetScalar(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n===== Training {target_name} model on {DEVICE} =====")
    print(model)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"[{target_name}] Epoch {epoch}/{EPOCHS} (train)", leave=False)
        for xb, yb in train_bar:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

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

        val_bar = tqdm(val_loader, desc=f"[{target_name}] Epoch {epoch}/{EPOCHS} (val)  ", leave=False)
        with torch.no_grad():
            for xb, yb in val_bar:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

                all_true.append(yb.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

        val_loss /= len(val_ds)
        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)

        mae = mean_absolute_error(all_true, all_pred)
        rmse = mean_squared_error(all_true, all_pred, squared=False)

        print(
            f"[{target_name}] Epoch {epoch:02d} | "
            f"Train MSE={train_loss:.3f} | Val MSE={val_loss:.3f} | "
            f"MAE={mae:.2f} | RMSE={rmse:.2f}"
        )

    # Save model
    os.makedirs("deep_bp_artifacts", exist_ok=True)
    model_path = os.path.join("deep_bp_artifacts", f"ppg_lstm_{target_name.lower()}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[{target_name}] Saved model to {model_path}")

    return model

def main():
    print("Building dataset from split records...")
    X, y = build_dataset_from_split(SPLIT_DIR, FS, MAX_CYCLE_LEN)
    sbp_all = y[:, 0]
    dbp_all = y[:, 1]

    # Train SBP model
    model_sbp = train_single_target(X, sbp_all, target_name="SBP")

    # Train DBP model
    model_dbp = train_single_target(X, dbp_all, target_name="DBP")

if __name__ == "__main__":
    main()
