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

    plot_windows_stacked(windows, meta)


if __name__ == "__main__":
    main()
