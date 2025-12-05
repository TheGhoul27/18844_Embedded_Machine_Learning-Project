#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn


# ==================== PREPROCESSING (same idea as before) ====================

def preprocess_arduino_csv_to_windows(
    csv_path,
    fs_target=125.0,
    win_sec=8.0,
    hop_sec=4.0,
    lowcut=0.5,
    highcut=5.0,
):
    """
    Load Arduino PPG CSV (millis, ir, red, green),
    resample IR to fs_target, band-pass, normalize,
    and return overlapping windows.
    """
    df = pd.read_csv(csv_path)

    # 1) Extract time + IR
    t_ms = df["millis"].to_numpy().astype(float)
    ir   = df["ir"].to_numpy().astype(float)

    # Convert to seconds starting at 0
    t_sec = (t_ms - t_ms[0]) / 1000.0

    # Estimate original Fs from timestamps
    dt = np.median(np.diff(t_sec))
    fs_orig = 1.0 / dt

    # 2) Resample to uniform grid at fs_target
    duration = t_sec[-1] - t_sec[0]
    n_target = int(np.floor(duration * fs_target))
    if n_target <= 0:
        raise ValueError("Not enough data to resample.")

    t_uniform = np.linspace(0, duration, n_target, endpoint=False)

    # Resample IR to uniform grid
    ir_resampled = np.interp(t_uniform, t_sec, ir)

    # 3) Band-pass filter
    nyq = 0.5 * fs_target
    if not (0 < lowcut < highcut < nyq):
        raise ValueError("Filter frequencies must satisfy 0 < lowcut < highcut < Nyquist.")

    b, a = butter(3, [lowcut / nyq, highcut / nyq], btype="band")

    ir_detrend = ir_resampled - np.mean(ir_resampled)
    ir_filt = filtfilt(b, a, ir_detrend)
    ir_filt = -ir_filt  # invert to have positive pulses

    # 4) Windowing
    win_len = int(win_sec * fs_target)
    hop_len = int(hop_sec * fs_target)

    windows = []
    win_start_times = []

    i = 0
    while i + win_len <= len(ir_filt):
        w = ir_filt[i : i + win_len]

        # 5) Per-window z-score
        w_mean = np.mean(w)
        w_std = np.std(w)
        if w_std < 1e-6:
            i += hop_len
            continue

        w_norm = (w - w_mean) / w_std
        windows.append(w_norm)
        win_start_times.append(t_uniform[i])
        i += hop_len

    windows = np.stack(windows, axis=0) if windows else np.empty((0, win_len))

    meta = {
        "fs_target": fs_target,
        "win_sec": win_sec,
        "hop_sec": hop_sec,
        "lowcut": lowcut,
        "highcut": highcut,
        "fs_original_est": fs_orig,
        "window_start_times": np.array(win_start_times),
    }

    return windows, meta


# ==================== MODEL (same as training) ====================

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
        weights = torch.softmax(score, dim=1)   # (B, T, 1)
        context = (weights * h).sum(dim=1)      # (B, H)
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
        h = self.conv(x)       # (B, C, T_cnn)
        h = h.transpose(1, 2)  # (B, T_cnn, C)

        lstm_out, _ = self.lstm(h)                 # (B, T_cnn, H_out)
        context, attn_weights = self.attention(lstm_out)  # (B, H_out)

        out = self.fc(context)                     # (B, 2) standardized
        return out


def load_trained_model(model_path, device):
    """
    Load best_state from training:
      { 'model': state_dict, 'y_mean': np.array(2,), 'y_std': np.array(2,) }
    """
    state = torch.load(model_path, map_location=device)

    model = CNNBiLSTMAttentionRegressor().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    y_mean = state["y_mean"].astype(np.float32)
    y_std  = state["y_std"].astype(np.float32)

    return model, y_mean, y_std


# ==================== INFERENCE & VISUALIZATION ====================

def predict_windows(windows, model, y_mean, y_std, device):
    """
    windows: (N, T)
    returns: preds_real (N, 2) with SBP, DBP in mmHg
    """
    if windows.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    x = torch.from_numpy(windows.astype(np.float32)).unsqueeze(1)  # (N,1,T)
    x = x.to(device)

    with torch.no_grad():
        preds_std = model(x).cpu().numpy()          # standardized
    preds_real = preds_std * y_std + y_mean         # de-standardize

    return preds_real


def plot_windows_with_preds(windows, meta, preds):
    """
    Plot each window stacked, with SBP/DBP prediction in the title.
    """
    num_windows, win_len = windows.shape
    if num_windows == 0:
        print("No windows to visualize.")
        return

    fig_height = 2.0 * num_windows
    plt.figure(figsize=(8, fig_height))

    for idx in range(num_windows):
        w = windows[idx]
        start_t = meta["window_start_times"][idx]
        sbp, dbp = preds[idx]

        ax = plt.subplot(num_windows, 1, idx + 1)
        ax.plot(w)
        ax.set_ylabel("Norm PPG")
        ax.set_title(
            f"Win {idx+1} (t={start_t:.2f}s)  "
            f"Pred SBP={sbp:.1f} mmHg, DBP={dbp:.1f} mmHg"
        )

        if idx == num_windows - 1:
            ax.set_xlabel("Sample index")
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run BP inference on Arduino PPG CSV using trained CNN+BiLSTM+Attention model."
    )
    parser.add_argument("csv_path", type=str, help="Path to Arduino CSV file")
    parser.add_argument(
        "--model-path",
        type=str,
        default="deep_bp_artifacts/ppg_cnn_bilstm_sbp_dbp.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, do not visualize windows, only print predictions summary.",
    )
    parser.add_argument("--fs-target", type=float, default=125.0)
    parser.add_argument("--win-sec", type=float, default=8.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=5.0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Preprocess CSV → windows
    windows, meta = preprocess_arduino_csv_to_windows(
        csv_path=args.csv_path,
        fs_target=args.fs_target,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        lowcut=args.lowcut,
        highcut=args.highcut,
    )
    print(f"Original Fs ≈ {meta['fs_original_est']:.2f} Hz")
    print(f"Num windows: {windows.shape[0]}, length: {windows.shape[1]} samples")

    if windows.shape[0] == 0:
        print("No valid windows extracted. Exiting.")
        return

    # 2) Load trained model + y_mean/std
    model, y_mean, y_std = load_trained_model(args.model_path, device)
    print("Loaded model from:", args.model_path)

    # 3) Predict
    preds = predict_windows(windows, model, y_mean, y_std, device)

    # Simple summary: per-window and overall mean
    for i, (sbp, dbp) in enumerate(preds):
        t0 = meta["window_start_times"][i]
        print(f"Window {i+1:02d} (t={t0:6.2f}s): SBP={sbp:6.1f}  DBP={dbp:6.1f}")

    mean_sbp = preds[:, 0].mean()
    mean_dbp = preds[:, 1].mean()
    print(f"\nOverall mean prediction across windows: SBP={mean_sbp:.1f}, DBP={mean_dbp:.1f} mmHg")

    # 4) Optional visualization
    if not args.no_plot:
        plot_windows_with_preds(windows, meta, preds)


if __name__ == "__main__":
    main()

# With alot of pre-processing which messes the predictions
# #!/usr/bin/env python
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from scipy.signal import butter, filtfilt, wiener
# from scipy.ndimage import uniform_filter1d

# import torch
# import torch.nn as nn


# # ==================== PREPROCESSING (MATCHES preprocessing.py) ====================

# def butter_highpass(cut, fs, order=2):
#     nyq = 0.5 * fs
#     b, a = butter(order, cut / nyq, btype="high")
#     return b, a

# def butter_bandpass(lo, hi, fs, order=2):
#     nyq = 0.5 * fs
#     b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
#     return b, a


# def preprocess_arduino_csv_to_windows(
#     csv_path,
#     fs_target=125.0,
#     win_sec=8.0,
#     hop_sec=4.0,
#     # processing params
#     hp_cut=0.05,       # high-pass to remove drift
#     bp_lo=0.5,
#     bp_hi=8.0,         # match MIMIC PPG band
#     wiener_size=7,     # Wiener smoothing window (samples)
#     clip_stds=5.0,     # clip big spikes
# ):
#     """
#     Load Arduino PPG CSV (millis, ir, red, green), resample IR to fs_target,
#     apply Wiener + high-pass + band-pass + inversion + smoothing + clipping,
#     then window and z-score.
#     """
#     df = pd.read_csv(csv_path)

#     # ---- 1) time + IR ----
#     t_ms = df["millis"].to_numpy().astype(float)
#     ir   = df["ir"].to_numpy().astype(float)

#     t_sec = (t_ms - t_ms[0]) / 1000.0
#     dt = np.median(np.diff(t_sec))
#     fs_orig = 1.0 / dt

#     # ---- 2) resample to uniform grid at fs_target ----
#     duration = t_sec[-1] - t_sec[0]
#     n_target = int(np.floor(duration * fs_target))
#     if n_target <= 0:
#         raise ValueError("Not enough data to resample.")

#     t_uniform = np.linspace(0.0, duration, n_target, endpoint=False)
#     ir_resampled = np.interp(t_uniform, t_sec, ir)

#     # ---- 3) Wiener smoothing ----
#     ir_smooth = wiener(ir_resampled, mysize=wiener_size)

#     # ---- 4) High-pass to remove slow drift ----
#     b_hp, a_hp = butter_highpass(hp_cut, fs_target, order=2)
#     ir_hp = filtfilt(b_hp, a_hp, ir_smooth)

#     # ---- 5) Band-pass in heart-rate band (0.5–8 Hz) ----
#     b_bp, a_bp = butter_bandpass(bp_lo, bp_hi, fs_target, order=2)
#     ir_bp = filtfilt(b_bp, a_bp, ir_hp)

#     # Flip polarity so pulses go upward (to match MIMIC)
#     ir_bp = -ir_bp

#     # Additional gentle smoothing (like "soft" low-pass)
#     ir_bp = uniform_filter1d(ir_bp, size=wiener_size)

#     # ---- 6) Clip extreme spikes (motion artefacts) ----
#     mu = np.mean(ir_bp)
#     sigma = np.std(ir_bp) + 1e-6
#     ir_clipped = np.clip(ir_bp, mu - clip_stds * sigma, mu + clip_stds * sigma)

#     # ---- 7) Windowing + per-window z-score ----
#     win_len = int(win_sec * fs_target)
#     hop_len = int(hop_sec * fs_target)

#     windows = []
#     win_start_times = []
#     i = 0
#     while i + win_len <= len(ir_clipped):
#         w = ir_clipped[i : i + win_len]

#         w_mean = np.mean(w)
#         w_std = np.std(w)
#         if w_std < 1e-6:
#             i += hop_len
#             continue

#         w_norm = (w - w_mean) / w_std
#         windows.append(w_norm)
#         win_start_times.append(t_uniform[i])
#         i += hop_len

#     windows = np.stack(windows, axis=0) if windows else np.empty((0, win_len))

#     meta = {
#         "fs_target": fs_target,
#         "fs_original_est": fs_orig,
#         "win_sec": win_sec,
#         "hop_sec": hop_sec,
#         "t_uniform": t_uniform,
#         "window_start_times": np.array(win_start_times),
#         "hp_cut": hp_cut,
#         "bp_lo": bp_lo,
#         "bp_hi": bp_hi,
#         "wiener_size": wiener_size,
#         "clip_stds": clip_stds,
#     }

#     return windows, meta


# # ==================== MODEL (same as training) ====================

# class AttentionBlock(nn.Module):
#     """
#     Additive attention over sequence: produces a weighted sum of LSTM outputs.
#     input:  (B, T, H)
#     output: (B, H)
#     """
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.W = nn.Linear(hidden_dim, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1, bias=False)

#     def forward(self, h):
#         score = self.v(torch.tanh(self.W(h)))   # (B, T, 1)
#         weights = torch.softmax(score, dim=1)   # (B, T, 1)
#         context = (weights * h).sum(dim=1)      # (B, H)
#         return context, weights


# class CNNBiLSTMAttentionRegressor(nn.Module):
#     """
#     CNN + BiLSTM + Additive Attention → regression head.
#     Output = standardized [SBP, DBP].
#     """
#     def __init__(self,
#                  num_conv_channels=64,
#                  lstm_hidden=64,
#                  lstm_layers=1,
#                  bidirectional=True,
#                  dropout=0.3):
#         super().__init__()

#         # CNN feature extractor
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(32, num_conv_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),

#             nn.Conv1d(num_conv_channels, num_conv_channels, kernel_size=5, padding=2),
#             nn.ReLU(),
#         )

#         self.bidirectional = bidirectional
#         num_dirs = 2 if bidirectional else 1
#         lstm_input_size = num_conv_channels
#         lstm_output_size = lstm_hidden * num_dirs

#         # BiLSTM
#         self.lstm = nn.LSTM(
#             input_size=lstm_input_size,
#             hidden_size=lstm_hidden,
#             num_layers=lstm_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#             dropout=dropout if lstm_layers > 1 else 0.0,
#         )

#         # Attention
#         self.attention = AttentionBlock(lstm_output_size)

#         # Final regression head
#         self.fc = nn.Sequential(
#             nn.Linear(lstm_output_size, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 2)
#         )

#     def forward(self, x):
#         # x: (B, 1, T_raw)
#         h = self.conv(x)       # (B, C, T_cnn)
#         h = h.transpose(1, 2)  # (B, T_cnn, C)

#         lstm_out, _ = self.lstm(h)                 # (B, T_cnn, H_out)
#         context, attn_weights = self.attention(lstm_out)  # (B, H_out)

#         out = self.fc(context)                     # (B, 2) standardized
#         return out


# def load_trained_model(model_path, device):
#     """
#     Load best_state from training:
#       { 'model': state_dict, 'y_mean': np.array(2,), 'y_std': np.array(2,) }
#     """
#     state = torch.load(model_path, map_location=device)

#     model = CNNBiLSTMAttentionRegressor().to(device)
#     model.load_state_dict(state["model"])
#     model.eval()

#     y_mean = state["y_mean"].astype(np.float32)
#     y_std  = state["y_std"].astype(np.float32)

#     return model, y_mean, y_std


# # ==================== INFERENCE & VISUALIZATION ====================

# def predict_windows(windows, model, y_mean, y_std, device):
#     """
#     windows: (N, T)
#     returns: preds_real (N, 2) with SBP, DBP in mmHg
#     """
#     if windows.shape[0] == 0:
#         return np.empty((0, 2), dtype=np.float32)

#     x = torch.from_numpy(windows.astype(np.float32)).unsqueeze(1)  # (N,1,T)
#     x = x.to(device)

#     with torch.no_grad():
#         preds_std = model(x).cpu().numpy()          # standardized
#     preds_real = preds_std * y_std + y_mean         # de-standardize

#     return preds_real


# def plot_windows_with_preds(windows, meta, preds):
#     """
#     Plot each window stacked, with SBP/DBP prediction in the title.
#     """
#     num_windows, win_len = windows.shape
#     if num_windows == 0:
#         print("No windows to visualize.")
#         return

#     fig_height = 2.0 * num_windows
#     plt.figure(figsize=(8, fig_height))

#     for idx in range(num_windows):
#         w = windows[idx]
#         start_t = meta["window_start_times"][idx]
#         sbp, dbp = preds[idx]

#         ax = plt.subplot(num_windows, 1, idx + 1)
#         ax.plot(w)
#         ax.set_ylabel("Norm PPG")
#         ax.set_title(
#             f"Win {idx+1} (t={start_t:.2f}s)  "
#             f"Pred SBP={sbp:.1f} mmHg, DBP={dbp:.1f} mmHg"
#         )

#         if idx == num_windows - 1:
#             ax.set_xlabel("Sample index")
#         else:
#             ax.set_xticklabels([])

#     plt.tight_layout()
#     plt.show()


# def main():
#     parser = argparse.ArgumentParser(
#         description="Run BP inference on Arduino PPG CSV using trained CNN+BiLSTM+Attention model."
#     )
#     parser.add_argument("csv_path", type=str, help="Path to Arduino CSV file")
#     parser.add_argument(
#         "--model-path",
#         type=str,
#         default="deep_bp_artifacts/ppg_cnn_bilstm_sbp_dbp.pt",
#         help="Path to trained model checkpoint",
#     )
#     parser.add_argument(
#         "--no-plot",
#         action="store_true",
#         help="If set, do not visualize windows, only print predictions summary.",
#     )
#     parser.add_argument("--fs-target", type=float, default=125.0)
#     parser.add_argument("--win-sec", type=float, default=8.0)
#     parser.add_argument("--hop-sec", type=float, default=4.0)
#     parser.add_argument("--lowcut", type=float, default=0.5)
#     parser.add_argument("--highcut", type=float, default=8.0)
#     parser.add_argument("--hp-cut", type=float, default=0.05)
#     parser.add_argument("--wiener-size", type=int, default=7)
#     parser.add_argument("--clip-stds", type=float, default=5.0)

#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # 1) Preprocess CSV → windows
#     windows, meta = preprocess_arduino_csv_to_windows(
#         csv_path=args.csv_path,
#         fs_target=args.fs_target,
#         win_sec=args.win_sec,
#         hop_sec=args.hop_sec,
#         hp_cut=args.hp_cut,
#         bp_lo=args.lowcut,
#         bp_hi=args.highcut,
#         wiener_size=args.wiener_size,
#         clip_stds=args.clip_stds,
#     )
#     print(f"Original Fs ≈ {meta['fs_original_est']:.2f} Hz")
#     print(f"Num windows: {windows.shape[0]}, length: {windows.shape[1]} samples")

#     if windows.shape[0] == 0:
#         print("No valid windows extracted. Exiting.")
#         return

#     # 2) Load trained model + y_mean/std
#     model, y_mean, y_std = load_trained_model(args.model_path, device)
#     print("Loaded model from:", args.model_path)

#     # 3) Predict
#     preds = predict_windows(windows, model, y_mean, y_std, device)

#     # Simple summary: per-window and overall mean
#     for i, (sbp, dbp) in enumerate(preds):
#         t0 = meta["window_start_times"][i]
#         print(f"Window {i+1:02d} (t={t0:6.2f}s): SBP={sbp:6.1f}  DBP={dbp:6.1f}")

#     mean_sbp = preds[:, 0].mean()
#     mean_dbp = preds[:, 1].mean()
#     print(f"\nOverall mean prediction across windows: SBP={mean_sbp:.1f}, DBP={mean_dbp:.1f} mmHg")

#     # 4) Optional visualization
#     if not args.no_plot:
#         plot_windows_with_preds(windows, meta, preds)


# if __name__ == "__main__":
#     main()
