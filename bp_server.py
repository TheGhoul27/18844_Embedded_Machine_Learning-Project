#!/usr/bin/env python3
import time
import warnings

import numpy as np
from scipy.signal import butter, filtfilt

import qwiic_max3010x
import onnxruntime as ort

from flask import Flask, jsonify, render_template_string

warnings.filterwarnings("ignore")

# ================== CONFIG ==================

DURATION_SEC = 30.0     # recording time

FS_TARGET = 125.0
WIN_SEC   = 8.0
HOP_SEC   = 4.0

LOWCUT        = 0.5
HIGHCUT       = 5.0
FILTER_ORDER  = 3

ONNX_PATH     = "bp_model.onnx"
Y_MEAN_PATH   = "bp_y_mean.npy"
Y_STD_PATH    = "bp_y_std.npy"


# ================== FILTER + MODEL ==================

def make_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError("Filter frequencies must satisfy 0 < lowcut < highcut < Nyquist.")
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


b_bp, a_bp = make_bandpass(LOWCUT, HIGHCUT, FS_TARGET, FILTER_ORDER)


def load_onnx_and_stats(onnx_path, y_mean_path, y_std_path):
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    y_mean = np.load(y_mean_path).astype(np.float32)
    y_std  = np.load(y_std_path).astype(np.float32)
    return sess, y_mean, y_std


# Load once at startup
sess, y_mean, y_std = load_onnx_and_stats(ONNX_PATH, Y_MEAN_PATH, Y_STD_PATH)


# ================== AGGREGATION (TRIM + WEIGHT) ==================

def aggregate_bp_trimmed_weighted(sbp_vals, dbp_vals, win_quality, trim_quantile=0.15):
    """
    1) Trim SBP outliers based on quantiles (e.g., drop lowest & highest 15%).
    2) Compute a weighted mean using win_quality (e.g., per-window std) as weights.
    Same mask is applied to SBP and DBP so windows stay aligned.
    """
    sbp_vals = np.asarray(sbp_vals, dtype=float)
    dbp_vals = np.asarray(dbp_vals, dtype=float)
    win_quality = np.asarray(win_quality, dtype=float)

    n = sbp_vals.size
    if n == 0:
        raise ValueError("No window predictions to aggregate.")

    # --- Step 1: trim out SBP outliers ---
    if n >= 4:
        lo = np.quantile(sbp_vals, trim_quantile)
        hi = np.quantile(sbp_vals, 1.0 - trim_quantile)
        mask = (sbp_vals >= lo) & (sbp_vals <= hi)
        # If trimming killed too many windows, fall back to no trimming
        if mask.sum() < 2:
            mask = np.ones_like(sbp_vals, dtype=bool)
    else:
        mask = np.ones_like(sbp_vals, dtype=bool)

    sbp_core = sbp_vals[mask]
    dbp_core = dbp_vals[mask]
    q_core   = win_quality[mask]

    # --- Step 2: weight by quality (std amplitude) ---
    q_core = np.clip(q_core, 1e-6, None)  # avoid zeros
    weights = q_core / q_core.sum()

    mean_sbp = float(np.sum(weights * sbp_core))
    mean_dbp = float(np.sum(weights * dbp_core))

    return mean_sbp, mean_dbp


# ================== CORE PIPELINE ==================

def preprocess_time_ir_to_windows(
    t_list,
    ir_list,
    fs_target=FS_TARGET,
    win_sec=WIN_SEC,
    hop_sec=HOP_SEC,
):
    t = np.array(t_list, dtype=float)
    ir = np.array(ir_list, dtype=float)
    ir = -ir

    if t.size < 2:
        raise ValueError("Not enough samples recorded.")

    # Reset time to start at 0
    t = t - t[0]
    dt = np.median(np.diff(t))
    fs_orig = 1.0 / dt if dt > 0 else 0.0

    duration = t[-1] - t[0]
    if duration < win_sec:
        raise ValueError(f"Recording too short: {duration:.2f}s (need at least {win_sec}s).")

    n_target = int(np.floor(duration * fs_target))
    if n_target <= 0:
        raise ValueError("Not enough data to resample.")

    # Uniform resample
    t_uniform = np.linspace(0.0, duration, n_target, endpoint=False)
    ir_resampled = np.interp(t_uniform, t, ir)

    # Detrend + band-pass
    ir_detrend = ir_resampled - np.mean(ir_resampled)
    ir_filt = filtfilt(b_bp, a_bp, ir_detrend)
    # ir_filt = -ir_filt  # invert so pulses are positive like MIMIC

    win_len = int(win_sec * fs_target)
    hop_len = int(hop_sec * fs_target)

    windows = []
    start_times = []
    win_stds = []  # per-window std as a simple quality metric

    i = 0
    while i + win_len <= len(ir_filt):
        w = ir_filt[i:i + win_len]
        m = np.mean(w)
        s = np.std(w)
        if s >= 1e-6:
            w_norm = (w - m) / s
            windows.append(w_norm.astype(np.float32))
            start_times.append(i / fs_target)
            win_stds.append(float(s))
        i += hop_len

    if not windows:
        raise ValueError("No valid windows extracted (signal too flat).")

    windows = np.stack(windows, axis=0)
    start_times = np.array(start_times)
    win_stds = np.array(win_stds)

    meta = {
        "fs_target": fs_target,
        "win_sec": win_sec,
        "hop_sec": hop_sec,
        "fs_original_est": fs_orig,
        "duration": duration,
        "t_uniform": t_uniform,
        "win_stds": win_stds,
    }
    extra = {
        "t_uniform": t_uniform,
        "ir_resampled": ir_resampled,
        "ir_filtered": ir_filt,
    }
    return windows, start_times, meta, extra


def predict_windows_onnx(windows):
    if windows.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = windows.astype(np.float32)[:, None, :]  # (N, 1, T)
    inputs = {"x": x}
    y_std_out = sess.run(["y"], inputs)[0]
    preds_real = y_std_out * y_std + y_mean
    return preds_real


def init_sensor():
    sensor = qwiic_max3010x.QwiicMax3010x()
    if sensor.begin() == False:
        raise RuntimeError("MAX3010x not detected. Check wiring/I2C.")
    sensor.setup(
        powerLevel=0x1F,
        sampleAverage=2,
        ledMode=3,
        sampleRate=200,
        pulseWidth=411,
        adcRange=8192,
    )
    return sensor


def record_ppg(sensor, duration_sec):
    t0 = time.time()
    t_list = []
    ir_list = []
    while True:
        now = time.time()
        if now - t0 >= duration_sec:
            break
        ir_val = sensor.getIR()
        sensor.nextSample()
        t_rel = now - t0
        t_list.append(t_rel)
        ir_list.append(float(ir_val))
    return t_list, ir_list


def run_full_measurement(duration_sec):
    sensor = init_sensor()
    t_list, ir_list = record_ppg(sensor, duration_sec)
    if not ir_list:
        raise RuntimeError("No samples recorded.")
    windows, start_times, meta, extra = preprocess_time_ir_to_windows(
        t_list, ir_list, fs_target=FS_TARGET, win_sec=WIN_SEC, hop_sec=HOP_SEC
    )
    preds_real = predict_windows_onnx(windows)
    sbp_vals = preds_real[:, 0]
    dbp_vals = preds_real[:, 1]

    return {
        "meta": meta,
        "extra": extra,
        "start_times": start_times,
        "sbp_vals": sbp_vals,
        "dbp_vals": dbp_vals,
    }


# ================== FLASK APP + FRONTEND ==================

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cuffless BP Demo</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin-top: 0;
      font-weight: 600;
      color: #f9fafb;
    }
    .card {
      background: #020617;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
      margin-bottom: 20px;
    }
    button {
      background: #2563eb;
      color: white;
      border: none;
      padding: 10px 18px;
      border-radius: 999px;
      font-size: 15px;
      cursor: pointer;
      font-weight: 500;
    }
    button:disabled {
      background: #1e293b;
      cursor: default;
    }
    .secondary-btn {
      background: #0b1120;
      border: 1px solid #1f2937;
      margin-left: 8px;
    }
    .status {
      margin-top: 8px;
      font-size: 14px;
      color: #9ca3af;
    }
    .metrics {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .metric {
      flex: 1;
      min-width: 120px;
      background: #020617;
      border-radius: 12px;
      padding: 12px 14px;
      text-align: center;
      border: 1px solid #1f2937;
    }
    .metric-label {
      font-size: 13px;
      color: #9ca3af;
      margin-bottom: 4px;
    }
    .metric-value {
      font-size: 22px;
      font-weight: 600;
      color: #f9fafb;
    }

    /* ===== Loading / heartbeat area ===== */
    .loading-area {
      margin-top: 16px;
      text-align: center;
    }
    .heart-loader {
      font-size: 54px;
      animation: beat 1s infinite;
      display: inline-block;
    }
    .pulse-line {
      margin: 10px auto 4px auto;
      width: 240px;
      height: 3px;
      background: linear-gradient(to right, #0f172a, #334155, #0f172a);
      position: relative;
      overflow: hidden;
      border-radius: 999px;
    }
    .pulse-dot {
      position: absolute;
      top: -4px;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: #ef4444;
      box-shadow: 0 0 12px rgba(248, 113, 113, 0.9);
      animation: pulse-run 1.4s linear infinite;
    }
    .loading-caption {
      font-size: 13px;
      color: #9ca3af;
      margin-top: 4px;
    }
    @keyframes beat {
      0%   { transform: scale(1);   }
      25%  { transform: scale(1.25);}
      40%  { transform: scale(1);   }
      60%  { transform: scale(1.18);}
      100% { transform: scale(1);   }
    }
    @keyframes pulse-run {
      0%   { left: -5%; }
      100% { left: 100%; }
    }

    canvas {
      background: #020617;
      border-radius: 12px;
      padding: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid #1f2937;
      padding: 6px 8px;
      text-align: right;
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      background: #020617;
      color: #9ca3af;
      position: sticky;
      top: 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🫀 Cuffless Blood Pressure Demo</h1>
    <div style="margin-top:-8px; margin-bottom:16px; color:#9ca3af; font-size:14px;">
      18-844 / 18-444 Embedded Machine Learning • Prof. Ziad Youssfi<br>
      Team: Pradhumna Guru Prasad, Om Kulkarni, Aidan Vogt
    </div>

    <div class="card">
      <p>Place your finger on the MAX30101 sensor and click <b>Start measurement</b>. You'll get a PPG waveform and SBP/DBP estimate.</p>
      <button id="startBtn">▶ Start measurement</button>
      <button id="resetZoomBtn" class="secondary-btn">Reset zoom</button>
      <div class="status" id="statusText">Idle. Ready when you are.</div>

      <div id="loadingArea" class="loading-area" style="display:none;">
        <div class="heart-loader">❤️</div>
        <div class="pulse-line">
          <div class="pulse-dot"></div>
        </div>
        <div class="loading-caption">
          Recording in progress… keep your finger steady on the sensor.
        </div>
      </div>
    </div>

    <div class="card">
      <h2 style="margin-top:0;">Results</h2>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Mean SBP</div>
          <div class="metric-value" id="sbpVal">--</div>
        </div>
        <div class="metric">
          <div class="metric-label">Mean DBP</div>
          <div class="metric-value" id="dbpVal">--</div>
        </div>
        <div class="metric">
          <div class="metric-label">Windows</div>
          <div class="metric-value" id="winCount">--</div>
        </div>
      </div>
      <div class="status" id="metaText"></div>
    </div>

    <div class="card">
      <h2 style="margin-top:0;">PPG Waveform (filtered)</h2>
      <canvas id="ppgChart" height="160"></canvas>
    </div>

    <div class="card">
      <h2 style="margin-top:0;">Per-window predictions</h2>
      <div style="max-height:220px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Start t (s)</th>
              <th>SBP (mmHg)</th>
              <th>DBP (mmHg)</th>
            </tr>
          </thead>
          <tbody id="windowTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
  const startBtn = document.getElementById('startBtn');
  const resetZoomBtn = document.getElementById('resetZoomBtn');
  const statusText = document.getElementById('statusText');
  const loadingArea = document.getElementById('loadingArea');
  const sbpVal = document.getElementById('sbpVal');
  const dbpVal = document.getElementById('dbpVal');
  const winCount = document.getElementById('winCount');
  const metaText = document.getElementById('metaText');
  const windowTableBody = document.getElementById('windowTableBody');

  let ppgChart = null;

  function clearChart() {
    if (ppgChart) {
      ppgChart.destroy();
      ppgChart = null;
    }
    const canvas = document.getElementById('ppgChart');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function setMeasuring(on) {
    if (on) {
      startBtn.disabled = true;
      statusText.textContent = "Measuring… keep your finger steady on the sensor. This may take around 30 seconds.";
      loadingArea.style.display = "block";
    } else {
      startBtn.disabled = false;
      loadingArea.style.display = "none";
    }
  }

  function updateChart(time, ppg) {
    const ctx = document.getElementById('ppgChart').getContext('2d');

    clearChart();

    ppgChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: time,
        datasets: [{
          label: 'PPG (filtered)',
          data: ppg,
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.2
        }]
      },
      options: {
        animation: false,
        plugins: {
          legend: { display: false },
          zoom: {
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              mode: 'x'
            },
            pan: {
              enabled: true,
              mode: 'x'
            }
          }
        },
        interaction: {
          mode: 'index',
          intersect: false
        },
        scales: {
          x: {
            title: { display: true, text: 'Time (s)' },
            ticks: { maxTicksLimit: 8 }
          },
          y: {
            title: { display: true, text: 'Amplitude (norm.)' }
          }
        }
      }
    });
  }

  startBtn.addEventListener('click', () => {
    setMeasuring(true);

    // Clear previous results (including chart) while we record
    clearChart();
    statusText.textContent = "Measuring… please wait.";
    windowTableBody.innerHTML = "";
    sbpVal.textContent = "--";
    dbpVal.textContent = "--";
    winCount.textContent = "--";
    metaText.textContent = "";

    fetch('/measure')
      .then(resp => {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(data => {
        setMeasuring(false);

        if (data.error) {
          statusText.textContent = "Error: " + data.error;
          return;
        }

        statusText.textContent = "Done. You can run another measurement anytime.";

        sbpVal.textContent = data.mean_sbp.toFixed(1) + " mmHg";
        dbpVal.textContent = data.mean_dbp.toFixed(1) + " mmHg";
        winCount.textContent = data.sbp_vals.length;

        metaText.textContent =
          "Original Fs ≈ " + data.fs_original_est.toFixed(1) + " Hz · " +
          "Resampled to " + data.fs_target.toFixed(0) + " Hz · " +
          "Duration ≈ " + data.duration.toFixed(1) + " s";

        updateChart(data.time_s, data.ppg_filtered);

        windowTableBody.innerHTML = "";
        data.sbp_vals.forEach((sbp, idx) => {
          const tr = document.createElement('tr');
          const start_t = data.start_times[idx];
          const dbp = data.dbp_vals[idx];
          tr.innerHTML =
            "<td>" + (idx + 1) + "</td>" +
            "<td>" + start_t.toFixed(1) + "</td>" +
            "<td>" + sbp.toFixed(1) + "</td>" +
            "<td>" + dbp.toFixed(1) + "</td>";
          windowTableBody.appendChild(tr);
        });
      })
      .catch(err => {
        setMeasuring(false);
        statusText.textContent = "Error: " + err;
      });
  });

  resetZoomBtn.addEventListener('click', () => {
    if (ppgChart && ppgChart.resetZoom) {
      ppgChart.resetZoom();
    }
  });
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/measure")
def measure():
    try:
        res = run_full_measurement(DURATION_SEC)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    meta = res["meta"]
    extra = res["extra"]
    start_times = res["start_times"]
    sbp_vals = res["sbp_vals"]
    dbp_vals = res["dbp_vals"]

    # Per-window quality (std of filtered PPG before normalization)
    win_stds = meta.get("win_stds", np.ones_like(sbp_vals))

    # downsample waveform a bit for plotting (keep it small-ish)
    t = extra["t_uniform"]
    ppg = extra["ir_filtered"]
    step = max(1, int(len(t) / 1000))  # up to ~1000 points
    t_ds = t[::step]
    ppg_ds = ppg[::step]

    # Trimmed + quality-weighted aggregation
    mean_sbp, mean_dbp = aggregate_bp_trimmed_weighted(sbp_vals, dbp_vals, win_stds)

    return jsonify({
        "mean_sbp": mean_sbp,
        "mean_dbp": mean_dbp,
        "fs_original_est": float(meta["fs_original_est"]),
        "fs_target": float(meta["fs_target"]),
        "duration": float(meta["duration"]),
        "time_s": t_ds.tolist(),
        "ppg_filtered": ppg_ds.tolist(),
        "start_times": start_times.tolist(),
        "sbp_vals": sbp_vals.tolist(),
        "dbp_vals": dbp_vals.tolist(),
    })


if __name__ == "__main__":
    # Bind to all interfaces so you can reach it via the Pi's IP
    app.run(host="0.0.0.0", port=8000)
