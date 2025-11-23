# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, find_peaks

# # ----------------- USER CONFIG -----------------

# NPZ_PATH = r"E:\\Masters_College_Work\\Semester_4\\Embedded_Deep_Learning\\18844_Embedded_Machine_Learning-Project\\split_records\\Part_1_group_0000.npz"
# RECORD_KEY = "record_2"
# FS = 125.0

# WINDOW_START = 0.0
# WINDOW_LEN   = 2.5
# # ------------------------------------------------

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

# # ------------- 1. LOAD + FILTER -------------

# data = np.load(NPZ_PATH, allow_pickle=True)
# rec = data[RECORD_KEY]
# ppg_raw = rec[:, 0].astype(float)
# abp_raw = rec[:, 1].astype(float)

# N = len(ppg_raw)
# t = np.arange(N) / FS

# b_ppg, a_ppg = butter_bandpass(0.5, 8.0, FS, order=2)
# ppg = apply_filter(ppg_raw, b_ppg, a_ppg)

# b_abp, a_abp = butter_lowpass(20.0, FS, order=2)
# abp = apply_filter(abp_raw, b_abp, a_abp)

# # ------------- 2. PPG peaks & troughs ---------

# min_distance = int(FS * 60.0 / 180.0)  # up to 180 bpm

# # PPG peaks (systolic-ish)
# ppg_peak_idx_all, _ = find_peaks(
#     ppg,
#     distance=min_distance,
#     prominence=0.2
# )

# # PPG troughs (valleys)
# ppg_trough_idx_all, _ = find_peaks(
#     -ppg,
#     distance=min_distance,
#     prominence=0.05
# )

# # ------------- 3. WINDOWED PLOT (context) -------------

# t0 = WINDOW_START
# t1 = WINDOW_START + WINDOW_LEN

# mask = (t >= t0) & (t <= t1)
# t_win   = t[mask]
# ppg_win = ppg[mask]
# abp_win = abp[mask]

# peaks_in_win   = ppg_peak_idx_all[(t[ppg_peak_idx_all]   >= t0) & (t[ppg_peak_idx_all]   <= t1)]
# troughs_in_win = ppg_trough_idx_all[(t[ppg_trough_idx_all] >= t0) & (t[ppg_trough_idx_all] <= t1)]

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# # PPG
# ax1.plot(t_win, ppg_win, label="PPG", linewidth=1.5)
# ax1.scatter(t[peaks_in_win], ppg[peaks_in_win],
#             color="C1", marker="o", s=60, label="PPG peaks", zorder=3)
# ax1.scatter(t[troughs_in_win], ppg[troughs_in_win],
#             color="C3", marker="v", s=60, label="PPG troughs", zorder=3)
# ax1.axvline(t0, color="k", linestyle="--", linewidth=0.8)
# ax1.axvline(t1, color="k", linestyle="--", linewidth=0.8)
# ax1.set_ylabel("PPG (a.u.)")
# ax1.legend(loc="upper right")
# ax1.set_title(f"PPG & ABP with PPG peaks & troughs\n"
#               f"{RECORD_KEY}, t ∈ [{t0:.1f}, {t1:.1f}] s")

# # ABP
# ax2.plot(t_win, abp_win, label="ABP", linewidth=1.5)
# ax2.scatter(t[peaks_in_win], abp[peaks_in_win],
#             color="C1", marker="o", s=60, label="PPG peaks (aligned)", zorder=3)
# ax2.scatter(t[troughs_in_win], abp[troughs_in_win],
#             color="C3", marker="v", s=60, label="PPG troughs (aligned)", zorder=3)
# ax2.axvline(t0, color="k", linestyle="--", linewidth=0.8)
# ax2.axvline(t1, color="k", linestyle="--", linewidth=0.8)
# ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("ABP (mmHg)")
# ax2.legend(loc="upper right")

# plt.tight_layout()
# plt.savefig("ppg_peaks_troughs_window.png", dpi=300)

# # ------------- 4. SINGLE-CYCLE PLOT (trough → trough) ---------

# # need at least two troughs to define one cycle
# if len(ppg_trough_idx_all) < 2:
#     raise RuntimeError("Not enough troughs to define a full cycle.")

# cycle_start = ppg_trough_idx_all[0]
# cycle_end   = ppg_trough_idx_all[1]

# t_cycle   = t[cycle_start:cycle_end+1]
# ppg_cycle = ppg[cycle_start:cycle_end+1]
# abp_cycle = abp[cycle_start:cycle_end+1]

# # find PPG peak inside THIS cycle
# local_peak = np.argmax(ppg_cycle)
# ppg_peak_t = t_cycle[local_peak]
# ppg_peak_v = ppg_cycle[local_peak]

# # cycle troughs (start & end)
# ppg_trough1_t = t[cycle_start]
# ppg_trough1_v = ppg[cycle_start]
# ppg_trough2_t = t[cycle_end]
# ppg_trough2_v = ppg[cycle_end]

# fig2, (bx1, bx2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

# # PPG cycle
# bx1.plot(t_cycle, ppg_cycle, label="PPG", linewidth=1.5)
# bx1.scatter(ppg_peak_t, ppg_peak_v, color="C1", marker="o", s=60, label="PPG peak")
# bx1.scatter([ppg_trough1_t, ppg_trough2_t],
#             [ppg_trough1_v, ppg_trough2_v],
#             color="C3", marker="v", s=60, label="PPG troughs")

# bx1.axvline(ppg_trough1_t, color="k", linestyle="--", linewidth=0.8)
# bx1.axvline(ppg_trough2_t, color="k", linestyle="--", linewidth=0.8)
# bx1.set_ylabel("PPG (a.u.)")
# bx1.legend(loc="upper right")
# bx1.set_title(f"Single PPG/ABP cycle (trough → trough)\n{RECORD_KEY}")

# # ABP cycle (aligned)
# bx2.plot(t_cycle, abp_cycle, label="ABP", linewidth=1.5)
# bx2.scatter(ppg_peak_t, abp_cycle[local_peak],
#             color="C1", marker="o", s=60, label="aligned PPG peak time")
# bx2.scatter([ppg_trough1_t, ppg_trough2_t],
#             [abp[cycle_start], abp[cycle_end]],
#             color="C3", marker="v", s=60, label="aligned PPG trough times")

# bx2.axvline(ppg_trough1_t, color="k", linestyle="--", linewidth=0.8)
# bx2.axvline(ppg_trough2_t, color="k", linestyle="--", linewidth=0.8)
# bx2.set_xlabel("Time (s)")
# bx2.set_ylabel("ABP (mmHg)")
# bx2.legend(loc="upper right")

# plt.tight_layout()
# plt.savefig("single_cycle_ppg_abp_trough_to_trough.png", dpi=300)
# # plt.show()

# # --- after you’ve already computed ppg, abp, t, etc. ---

# from scipy.signal import find_peaks
# import numpy as np
# import matplotlib.pyplot as plt

# # 1) detect systolic peaks on ABP and PPG
# min_distance = int(FS * 60.0 / 180.0)

# abp_peaks, _ = find_peaks(abp, distance=min_distance, prominence=10.0)
# ppg_peaks, _ = find_peaks(ppg, distance=min_distance, prominence=0.2)

# # 2) for each ABP peak, find nearest PPG peak AFTER it → estimate lag in samples
# lags = []
# for i in abp_peaks:
#     # PPG peak that occurs after this ABP peak
#     j_candidates = ppg_peaks[ppg_peaks > i]
#     if len(j_candidates) == 0:
#         continue
#     j = j_candidates[0]
#     lags.append(j - i)

# lags = np.array(lags)
# median_lag = int(np.median(lags))   # samples
# time_lag = median_lag / FS          # seconds
# print("Estimated ABP→PPG lag ~", time_lag, "s")

# # 3) shift PPG *earlier* by this lag for plotting only
# ppg_shifted = np.roll(ppg, -median_lag)

# # 4) plot a window with aligned cycles
# t0, t1 = 0.0, 2.5
# mask = (t >= t0) & (t <= t1)

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# ax1.plot(t[mask], abp[mask], label="ABP", linewidth=1.5)
# ax1.set_ylabel("ABP (mmHg)")
# ax1.legend(loc="upper right")

# ax2.plot(t[mask], ppg[mask], label="PPG original", alpha=0.4)
# ax2.plot(t[mask], ppg_shifted[mask], label="PPG shifted (aligned)", linewidth=1.5)
# ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("PPG (a.u.)")
# ax2.legend(loc="upper right")

# plt.tight_layout()
# plt.savefig("abp_ppg_aligned_by_lag.png", dpi=300)
# # plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ----------------- USER CONFIG -----------------

NPZ_PATH = r"E:\\Masters_College_Work\\Semester_4\\Embedded_Deep_Learning\\18844_Embedded_Machine_Learning-Project\\split_records\\Part_1_group_0000.npz"
RECORD_KEY = "record_1"
FS = 125.0

WINDOW_START = 0.0
WINDOW_LEN   = 10.0
# ------------------------------------------------

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

# ------------- 1. LOAD + FILTER -------------

data = np.load(NPZ_PATH, allow_pickle=True)
rec = data[RECORD_KEY]
ppg_raw = rec[:, 0].astype(float)
abp_raw = rec[:, 1].astype(float)

N = len(ppg_raw)
t = np.arange(N) / FS

b_ppg, a_ppg = butter_bandpass(0.5, 8.0, FS, order=2)
ppg = apply_filter(ppg_raw, b_ppg, a_ppg)

b_abp, a_abp = butter_lowpass(20.0, FS, order=2)
abp = apply_filter(abp_raw, b_abp, a_abp)

# ------------- 2. PPG peaks & troughs ---------

min_distance = int(FS * 60.0 / 180.0)  # up to 180 bpm

# PPG peaks (systolic-ish)
ppg_peak_idx_all, _ = find_peaks(
    ppg,
    distance=min_distance,
    prominence=0.2
)

# PPG troughs (valleys)
ppg_trough_idx_all, _ = find_peaks(
    -ppg,
    distance=min_distance,
    prominence=0.05
)

# ------------- 3. Estimate ABP→PPG lag from peaks ---------

abp_peaks, _ = find_peaks(abp, distance=min_distance, prominence=10.0)

lags = []
for i in abp_peaks:
    j_cand = ppg_peak_idx_all[ppg_peak_idx_all > i]
    if len(j_cand) == 0:
        continue
    j = j_cand[0]
    lags.append(j - i)

lags = np.array(lags)
median_lag = int(np.median(lags)) if len(lags) > 0 else 0
time_lag = median_lag / FS
print(f"Estimated ABP→PPG lag ≈ {time_lag:.4f} s ({median_lag} samples)")

# PPG shifted earlier (aligned to ABP) — for plotting only
ppg_aligned = np.roll(ppg, -median_lag)

# ------------- 4. Per-cycle SBP / DBP from trough→trough ---------

cycles = []  # list of dicts with indices and values

for k in range(len(ppg_trough_idx_all) - 1):
    start = ppg_trough_idx_all[k]
    end   = ppg_trough_idx_all[k+1]
    if end <= start:
        continue

    seg_abp = abp[start:end+1]
    sbp_local = np.argmax(seg_abp)
    dbp_local = np.argmin(seg_abp)

    sbp_idx = start + sbp_local
    dbp_idx = start + dbp_local

    cycles.append({
        "cycle_index": k,
        "start": start,
        "end": end,
        "sbp_idx": sbp_idx,
        "dbp_idx": dbp_idx,
        "sbp_val": abp[sbp_idx],
        "dbp_val": abp[dbp_idx],
    })

# Print a few cycles as sanity check
print("First few cycles (SBP/DBP):")
for c in cycles[:10]:
    print(
        f"Cycle {c['cycle_index']}: "
        f"SBP={c['sbp_val']:.2f} mmHg at t={t[c['sbp_idx']]:.3f}s, "
        f"DBP={c['dbp_val']:.2f} mmHg at t={t[c['dbp_idx']]:.3f}s"
    )

# ------------- 5. Define plotting window -------------

t0 = WINDOW_START
t1 = WINDOW_START + WINDOW_LEN
mask = (t >= t0) & (t <= t1)

t_win        = t[mask]
ppg_win      = ppg[mask]
ppg_aln_win  = ppg_aligned[mask]
abp_win      = abp[mask]

peaks_in_win   = ppg_peak_idx_all[(t[ppg_peak_idx_all]   >= t0) & (t[ppg_peak_idx_all]   <= t1)]
troughs_in_win = ppg_trough_idx_all[(t[ppg_trough_idx_all] >= t0) & (t[ppg_trough_idx_all] <= t1)]

# SBP/DBP indices that fall in window
sbp_idx_win = [c["sbp_idx"] for c in cycles if t0 <= t[c["sbp_idx"]] <= t1]
dbp_idx_win = [c["dbp_idx"] for c in cycles if t0 <= t[c["dbp_idx"]] <= t1]

# ------------- 6. Scale aligned PPG to ABP range for joint plot -------------

abp_min, abp_max = abp_win.min(), abp_win.max()
ppg_min, ppg_max = ppg_aln_win.min(), ppg_aln_win.max()

# avoid division by zero
ppg_scaled = (ppg_aln_win - ppg_min) / (ppg_max - ppg_min + 1e-9)
ppg_scaled = ppg_scaled * (abp_max - abp_min) + abp_min

# ------------- 7. Build one figure with 4 subplots -------------

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
ax1, ax2, ax3, ax4 = axes

# (1) PPG raw
ax1.plot(t_win, ppg_win, label="PPG", linewidth=1.5)
ax1.scatter(t[peaks_in_win], ppg[peaks_in_win],
            color="C1", marker="o", s=40, label="PPG peaks")
ax1.scatter(t[troughs_in_win], ppg[troughs_in_win],
            color="C3", marker="v", s=40, label="PPG troughs")
ax1.axvline(t0, color="k", linestyle="--", linewidth=0.8)
ax1.axvline(t1, color="k", linestyle="--", linewidth=0.8)
ax1.set_ylabel("PPG (a.u.)")
ax1.set_title(f"{RECORD_KEY}: PPG, aligned PPG, ABP, and SBP/DBP per cycle")
ax1.legend(loc="upper right")

# (2) PPG original vs aligned (both in PPG units)
ax2.plot(t_win, ppg_win, label="PPG original", alpha=0.5)
ax2.plot(t_win, ppg_aln_win, label=f"PPG aligned (shift ≈ {time_lag:.3f}s)", linewidth=1.5)
ax2.axvline(t0, color="k", linestyle="--", linewidth=0.8)
ax2.axvline(t1, color="k", linestyle="--", linewidth=0.8)
ax2.set_ylabel("PPG (a.u.)")
ax2.legend(loc="upper right")

# (3) ABP alone
ax3.plot(t_win, abp_win, label="ABP", linewidth=1.5)
ax3.axvline(t0, color="k", linestyle="--", linewidth=0.8)
ax3.axvline(t1, color="k", linestyle="--", linewidth=0.8)
ax3.set_ylabel("ABP (mmHg)")
ax3.legend(loc="upper right")

# (4) ABP + *scaled* aligned PPG + SBP/DBP markers
ax4.plot(t_win, abp_win, label="ABP", linewidth=1.5)
ax4.plot(t_win, ppg_scaled, label="PPG aligned (scaled to ABP)", alpha=0.8)

# SBP markers (on ABP)
ax4.scatter(t[sbp_idx_win], abp[sbp_idx_win],
            color="red", marker="^", s=60, label="SBP")
# DBP markers (on ABP)
ax4.scatter(t[dbp_idx_win], abp[dbp_idx_win],
            color="green", marker="v", s=60, label="DBP")

# optional value labels
for idx in sbp_idx_win:
    ax4.text(t[idx], abp[idx] + 1, f"{abp[idx]:.0f}", fontsize=7,
             ha="center", va="bottom")
for idx in dbp_idx_win:
    ax4.text(t[idx], abp[idx] - 3, f"{abp[idx]:.0f}", fontsize=7,
             ha="center", va="top")

ax4.axvline(t0, color="k", linestyle="--", linewidth=0.8)
ax4.axvline(t1, color="k", linestyle="--", linewidth=0.8)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("ABP / scaled PPG")
ax4.legend(loc="upper right")

for ax in axes:
    ax.set_xlim(t0, t1)

plt.tight_layout()
plt.savefig("combined_subplots_ppg_abp_sbp_dbp.png", dpi=300)
# plt.show()
