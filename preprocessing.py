#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


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
    ir_resampled = np.interp(t_uniform, t_sec, ir)

    # 3) Band-pass filter
    nyq = 0.5 * fs_target
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


def plot_windows_stacked(windows, meta):
    """
    Plot all windows stacked vertically with a tall figure,
    matching the look of the example image.
    """
    num_windows, win_len = windows.shape

    # This size is chosen to mimic your screenshot: tall and readable
    fig_height = 2.0 * num_windows  # 2 inches per window
    plt.figure(figsize=(8, fig_height))

    for idx in range(num_windows):
        w = windows[idx]
        start_t = meta["window_start_times"][idx]

        ax = plt.subplot(num_windows, 1, idx + 1)
        ax.plot(w)
        ax.set_ylabel("Norm PPG")
        ax.set_title(f"Window {idx+1} (Start at {start_t:.2f} s)")

        if idx == num_windows - 1:
            ax.set_xlabel("Sample index")
        else:
            # Hide x tick labels for all but last subplot to match clean stack
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.show()

def plot_windows_individual(windows, meta, save=False, prefix="window"):
    """
    Plot each window in its own figure (not subplots).
    Great for inspecting morphology without vertical squishing.
    If save=True, saves PNGs instead of showing interactively.
    """
    num_windows = windows.shape[0]

    for idx in range(num_windows):
        w = windows[idx]
        start_t = meta["window_start_times"][idx]

        plt.figure(figsize=(10, 4))  # wider + comfortable vertical spacing
        plt.plot(w, color="darkorange")
        plt.title(f"Window {idx+1} (Start at {start_t:.2f} s)")
        plt.xlabel("Sample index")
        plt.ylabel("Normalized PPG")
        plt.grid(alpha=0.3)

        if save:
            fname = f"{prefix}_{idx+1}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Arduino PPG CSV and visualize band-passed, normalized windows."
    )
    parser.add_argument("csv_path", type=str, help="Path to Arduino CSV file")
    parser.add_argument("--fs-target", type=float, default=125.0)
    parser.add_argument("--win-sec", type=float, default=8.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=5.0)

    args = parser.parse_args()

    windows, meta = preprocess_arduino_csv_to_windows(
        csv_path=args.csv_path,
        fs_target=args.fs_target,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        lowcut=args.lowcut,
        highcut=args.highcut,
    )

    print(f"Original Fs ≈ {meta['fs_original_est']:.2f} Hz")
    print(f"Num windows: {windows.shape[0]}, window length: {windows.shape[1]} samples")

    # plot_windows_stacked(windows, meta)
    plot_windows_individual(windows, meta)


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
#     apply Wiener + high-pass + band-pass + clipping, then window and z-score.
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

#     # ---- 3) Wiener smoothing (like the article) ----
#     # small window → denoise but keep pulse shape
#     ir_smooth = wiener(ir_resampled, mysize=wiener_size)

#     # ---- 4) High-pass to remove slow drift / motion baseline ----
#     b_hp, a_hp = butter_highpass(hp_cut, fs_target, order=2)
#     ir_hp = filtfilt(b_hp, a_hp, ir_smooth)

#     # ---- 5) Band-pass in heart-rate band (0.5–8 Hz) ----
#     b_bp, a_bp = butter_bandpass(bp_lo, bp_hi, fs_target, order=2)
#     ir_bp = filtfilt(b_bp, a_bp, ir_hp)
#     ir_bp = -ir_bp  # invert to have positive pulses
#     ir_bp = uniform_filter1d(ir_bp, size=wiener_size)

#     # ---- 6) Clip extreme spikes (e.g., movement) ----
#     mu = np.mean(ir_bp)
#     sigma = np.std(ir_bp) + 1e-6
#     ir_clipped = np.clip(ir_bp, mu - clip_stds * sigma, mu + clip_stds * sigma)

#     # ---- 7) Windowing ----
#     win_len = int(win_sec * fs_target)
#     hop_len = int(hop_sec * fs_target)

#     windows = []
#     win_start_times = []
#     i = 0
#     while i + win_len <= len(ir_clipped):
#         w = ir_clipped[i : i + win_len]

#         # per-window z-score
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


# def plot_windows_stacked(windows, meta):
#     """
#     Plot all windows stacked vertically with a tall figure,
#     matching the look of the example image.
#     """
#     num_windows, win_len = windows.shape

#     # This size is chosen to mimic your screenshot: tall and readable
#     fig_height = 2.0 * num_windows  # 2 inches per window
#     plt.figure(figsize=(8, fig_height))

#     for idx in range(num_windows):
#         w = windows[idx]
#         start_t = meta["window_start_times"][idx]

#         ax = plt.subplot(num_windows, 1, idx + 1)
#         ax.plot(w)
#         ax.set_ylabel("Norm PPG")
#         ax.set_title(f"Window {idx+1} (Start at {start_t:.2f} s)")

#         if idx == num_windows - 1:
#             ax.set_xlabel("Sample index")
#         else:
#             # Hide x tick labels for all but last subplot to match clean stack
#             ax.set_xticklabels([])

#     plt.tight_layout()
#     plt.show()

# def plot_windows_individual(windows, meta, save=False, prefix="window"):
#     """
#     Plot each window in its own figure (not subplots).
#     Great for inspecting morphology without vertical squishing.
#     If save=True, saves PNGs instead of showing interactively.
#     """
#     num_windows = windows.shape[0]

#     for idx in range(num_windows):
#         w = windows[idx]
#         start_t = meta["window_start_times"][idx]

#         plt.figure(figsize=(10, 4))  # wider + comfortable vertical spacing
#         plt.plot(w, color="darkorange")
#         plt.title(f"Window {idx+1} (Start at {start_t:.2f} s)")
#         plt.xlabel("Sample index")
#         plt.ylabel("Normalized PPG")
#         plt.grid(alpha=0.3)

#         if save:
#             fname = f"{prefix}_{idx+1}.png"
#             plt.savefig(fname, dpi=200, bbox_inches="tight")
#             plt.close()
#         else:
#             plt.show()


# def main():
#     parser = argparse.ArgumentParser(
#         description="Preprocess Arduino PPG CSV and visualize band-passed, normalized windows."
#     )
#     parser.add_argument("csv_path", type=str, help="Path to Arduino CSV file")
#     parser.add_argument("--fs-target", type=float, default=125.0)
#     parser.add_argument("--win-sec", type=float, default=8.0)
#     parser.add_argument("--hop-sec", type=float, default=4.0)
#     parser.add_argument("--lowcut", type=float, default=0.5)
#     parser.add_argument("--highcut", type=float, default=5.0)
#     parser.add_argument("--hp-cut", type=float, default=0.05)
#     parser.add_argument("--wiener-size", type=int, default=7)
#     parser.add_argument("--clip-stds", type=float, default=5.0)   

#     args = parser.parse_args()

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
#     print(f"Num windows: {windows.shape[0]}, window length: {windows.shape[1]} samples")

#     # plot_windows_stacked(windows, meta)
#     plot_windows_individual(windows, meta)


# if __name__ == "__main__":
#     main()
