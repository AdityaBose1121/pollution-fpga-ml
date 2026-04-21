"""
ZPHS01B + ESP32  →  PM2.5 Prediction Dashboard  (Flask backend)
================================================================
ESP32 POSTs JSON to  POST /api/reading
Dashboard polls      GET  /api/state
Dashboard served at  GET  /

Sensor features (ZPHS01B provides 5 of 9; remaining estimated):
  Index  Feature  Source
  -----  -------  ------
  0      PM10     ZPHS01B
  1      PM25     ZPHS01B
  2      PM1      ZPHS01B
  3      O3       ZPHS01B
  4      CO       estimated
  5      NO       estimated
  6      NO2      estimated
  7      NOx      estimated
  8      CO2      ZPHS01B
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import os
import json
import time
from collections import deque
from datetime import datetime, timezone
import threading

# ─────────────────────────────────────────────────────────────────────
# Paths  (server.py lives inside  .../FPGAvsPi/dashboard/)
# ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model_fpga_ready.h5")

# ─────────────────────────────────────────────────────────────────────
# Load model weights  (float32 via Keras → NumPy inference)
# ─────────────────────────────────────────────────────────────────────
WEIGHTS = []
BIASES  = []

def load_model_weights():
    global WEIGHTS, BIASES
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        for layer in model.layers:
            if "dense" in layer.name or "output" in layer.name:
                w, b = layer.get_weights()
                WEIGHTS.append(w.astype(np.float32))
                BIASES.append(b.astype(np.float32))
        print(f"[Model] Loaded {len(WEIGHTS)} layers from {MODEL_PATH}")
        for i, (w, b) in enumerate(zip(WEIGHTS, BIASES)):
            print(f"  Layer {i+1}: W{w.shape}  b{b.shape}")
    except Exception as e:
        print(f"[Model] WARNING – could not load Keras model: {e}")
        print("[Model] Falling back to integer weights from weights/ CSV files.")
        _load_csv_weights()

def _load_csv_weights():
    """Fallback: load fixed-point integer weights from CSV, apply scale=2^-7."""
    global WEIGHTS, BIASES
    SCALE = 2 ** -7          # Q1.7 fixed-point scale used during quantisation
    weights_dir = os.path.join(ROOT_DIR, "weights")
    for i in range(1, 4):
        W = np.loadtxt(os.path.join(weights_dir, f"W{i}.csv"), delimiter=",", dtype=np.float32) * SCALE
        b = np.loadtxt(os.path.join(weights_dir, f"b{i}.csv"), delimiter=",", dtype=np.float32) * SCALE
        WEIGHTS.append(W)
        BIASES.append(b)
    print(f"[Model] Loaded CSV weights (scaled by {SCALE})")

# ─────────────────────────────────────────────────────────────────────
# Training data statistics  (computed from y_targets.npy at startup)
# Used to detect when live inputs are out of the training distribution.
# ─────────────────────────────────────────────────────────────────────
TRAIN_STATS: dict = {}   # filled by load_training_stats()

# Per-feature stats from X_windows (shape N×54, features in groups of 9)
# We track PM25 (feature index 1 inside each 9-feature block) across all 6 steps
FEATURE_NAMES = ["PM10","PM25","PM1","O3","CO","NO","NO2","NOx","CO2"]
N_FEATURES = 9

def load_training_stats():
    """Load y_targets and X_windows to compute training distribution stats."""
    global TRAIN_STATS
    y_path = os.path.join(ROOT_DIR, "data", "y_targets.npy")
    x_path = os.path.join(ROOT_DIR, "data", "X_windows.npy")
    try:
        y = np.load(y_path).astype(np.float32)          # shape (N,)
        X = np.load(x_path).astype(np.float32)          # shape (N, 54)

        # PM25 target stats  (what the model was trained to predict)
        pm25_target_mean = float(np.mean(y))
        pm25_target_std  = float(np.std(y))
        pm25_target_min  = float(np.min(y))
        pm25_target_max  = float(np.max(y))

        # Per-feature input stats across all windows & timesteps
        # X[:,i::N_FEATURES] gives feature i across all 6 timesteps × N windows
        feature_stats = {}
        for fi, fname in enumerate(FEATURE_NAMES):
            cols = X[:, fi::N_FEATURES]   # shape (N, 6) — one column per time step
            vals = cols.flatten()
            feature_stats[fname] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
                "p5":   float(np.percentile(vals, 5)),
                "p95":  float(np.percentile(vals, 95)),
            }

        TRAIN_STATS = {
            "pm25_target": {
                "mean": pm25_target_mean,
                "std":  pm25_target_std,
                "min":  pm25_target_min,
                "max":  pm25_target_max,
            },
            "features": feature_stats,
            "n_samples": int(len(y)),
        }

        print(f"[Stats] Training stats loaded from {y_path}")
        print(f"  PM2.5 target — mean: {pm25_target_mean:.1f}  std: {pm25_target_std:.1f}  "
              f"range: [{pm25_target_min:.1f}, {pm25_target_max:.1f}] µg/m³")
        for fname in ["PM25", "CO2", "O3"]:
            s = feature_stats[fname]
            print(f"  {fname:6s} input  — mean: {s['mean']:.2f}  std: {s['std']:.2f}  "
                  f"p5–p95: [{s['p5']:.2f}, {s['p95']:.2f}]")
    except Exception as e:
        print(f"[Stats] WARNING – could not load training stats: {e}")
        # Fallback to approximate values derived from notebook outputs
        TRAIN_STATS = {
            "pm25_target": {"mean": 55.0, "std": 30.0, "min": 0.0, "max": 300.0},
            "features": {
                "PM25": {"mean": 55.0, "std": 30.0, "p5": 10.0, "p95": 120.0,
                         "min": 0.0, "max": 300.0},
            },
            "n_samples": 0,
        }

def model_warning(current_pm25_values: list[float]) -> dict:
    """
    Given the list of PM2.5 values in the current 6-hr window,
    return a warning dict describing how far they are from the training distribution.
    """
    if not TRAIN_STATS or not current_pm25_values:
        return {"level": "none", "message": "", "z_score": 0.0}

    feat_stats = TRAIN_STATS["features"].get("PM25", {})
    if not feat_stats:
        return {"level": "none", "message": "", "z_score": 0.0}

    mean = feat_stats["mean"]
    std  = feat_stats["std"] or 1.0
    p5   = feat_stats.get("p5", mean - 2*std)
    p95  = feat_stats.get("p95", mean + 2*std)

    avg_input = float(np.mean(current_pm25_values))
    z = (avg_input - mean) / std
    abs_z = abs(z)

    # Is the input inside the trained 5th–95th percentile range?
    in_range = p5 <= avg_input <= p95

    if in_range and abs_z < 1.5:
        return {
            "level":   "none",
            "message": "",
            "z_score": round(z, 2),
            "avg_input_pm25": round(avg_input, 1),
            "training_mean":  round(mean, 1),
            "training_p5":    round(p5, 1),
            "training_p95":   round(p95, 1),
        }
    elif abs_z < 2.5:
        direction = "lower" if z < 0 else "higher"
        return {
            "level":   "marginal",
            "message": (
                f"Current PM2.5 avg ({avg_input:.1f} µg/m³) is {direction} than the "
                f"training data mean ({mean:.1f} µg/m³). "
                f"Prediction may be less accurate."
            ),
            "z_score": round(z, 2),
            "avg_input_pm25": round(avg_input, 1),
            "training_mean":  round(mean, 1),
            "training_p5":    round(p5, 1),
            "training_p95":   round(p95, 1),
        }
    else:
        direction = "much lower" if z < 0 else "much higher"
        return {
            "level":   "severe",
            "message": (
                f"⚠ Out of Training Range: PM2.5 avg ({avg_input:.1f} µg/m³) is "
                f"{direction} than the training distribution "
                f"(mean {mean:.1f}, p5–p95: {p5:.1f}–{p95:.1f} µg/m³). "
                f"Prediction reliability is low."
            ),
            "z_score": round(z, 2),
            "avg_input_pm25": round(avg_input, 1),
            "training_mean":  round(mean, 1),
            "training_p5":    round(p5, 1),
            "training_p95":   round(p95, 1),
        }

# ─────────────────────────────────────────────────────────────────────
# NumPy MLP inference  (mirrors 04_reference_cpu_inference.ipynb)
# ─────────────────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def mlp_forward(x: np.ndarray) -> float:
    """x shape: (54,) or (1,54)"""
    x = np.array(x, dtype=np.float32).reshape(1, -1)
    z1 = x @ WEIGHTS[0] + BIASES[0]
    a1 = relu(z1)
    z2 = a1 @ WEIGHTS[1] + BIASES[1]
    a2 = relu(z2)
    z3 = a2 @ WEIGHTS[2] + BIASES[2]
    return float(z3[0, 0])

# ─────────────────────────────────────────────────────────────────────
# State: rolling 6-hour buffer
# ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 6
FEATURE_ORDER = ["PM10", "PM25", "PM1", "O3", "CO", "NO", "NO2", "NOx", "CO2"]

# Default estimates for sensors not on ZPHS01B
DEFAULTS = {"CO": 0.9, "NO": 3.0, "NO2": 6.0, "NOx": 9.0}

# Extra display-only fields forwarded by the ESP32 sketch
DISPLAY_EXTRAS = ["TEMP", "HUM", "CH2O", "VOC"]

history:  deque  = deque(maxlen=WINDOW_SIZE)   # each entry: {timestamp, features dict, prediction}
latest_reading: dict = {}
latest_prediction: float | None = None
lock = threading.Lock()

def make_feature_vector(reading: dict) -> np.ndarray:
    """Build 54-element float32 vector from a single reading dict."""
    vec = np.zeros(9, dtype=np.float32)
    for i, feat in enumerate(FEATURE_ORDER):
        if feat in reading:
            vec[i] = float(reading[feat])
        elif feat in DEFAULTS:
            vec[i] = DEFAULTS[feat]
    return vec

def run_prediction() -> float | None:
    """Run MLP on the current 6-entry window. Returns None if window not full."""
    if len(history) < WINDOW_SIZE:
        return None
    window_vecs = [make_feature_vector(h["reading"]) for h in list(history)]
    x = np.concatenate(window_vecs, axis=0)   # shape (54,)
    return mlp_forward(x)

def aqi_level(pm25: float) -> dict:
    """Return AQI category metadata for a PM2.5 value (µg/m³, 24-hr standard)."""
    if pm25 <= 12:
        return {"label": "Good",       "color": "#22c55e", "index": 1}
    elif pm25 <= 35.4:
        return {"label": "Moderate",   "color": "#eab308", "index": 2}
    elif pm25 <= 55.4:
        return {"label": "Unhealthy for Sensitive", "color": "#f97316", "index": 3}
    elif pm25 <= 150.4:
        return {"label": "Unhealthy",  "color": "#ef4444", "index": 4}
    elif pm25 <= 250.4:
        return {"label": "Very Unhealthy", "color": "#a855f7", "index": 5}
    else:
        return {"label": "Hazardous", "color": "#7f1d1d", "index": 6}

# ─────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/api/reading", methods=["POST"])
def post_reading():
    """
    ESP32 posts JSON like:
    {
      "PM1": 23.5, "PM25": 45.2, "PM10": 67.8,
      "CO2": 430.0, "O3": 12.0,
      "CO": 0.95, "NO": 3.1, "NO2": 5.9, "NOx": 9.0   ← optional
    }
    """
    global latest_reading, latest_prediction

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    ts = datetime.now(timezone.utc).isoformat()

    entry = {
        "timestamp": ts,
        "reading": data,
    }

    with lock:
        history.append(entry)
        latest_reading = data
        pred = run_prediction()
        latest_prediction = pred
        if pred is not None:
            history[-1]["prediction"] = pred

    return jsonify({"status": "ok", "timestamp": ts,
                    "buffer_size": len(history),
                    "prediction": pred})

@app.route("/api/state", methods=["GET"])
def get_state():
    """Dashboard polls this to refresh."""
    with lock:
        hist_list = list(history)
        pred      = latest_prediction
        current   = latest_reading.copy() if latest_reading else {}

    pm25_now  = float(current.get("PM25", 0))
    aqi_now   = aqi_level(pm25_now) if pm25_now else None
    aqi_pred  = aqi_level(float(pred)) if pred is not None else None

    # Build chart-friendly arrays
    chart_labels = [h["timestamp"] for h in hist_list]
    chart_pm25   = [float(h["reading"].get("PM25", 0)) for h in hist_list]
    chart_pm10   = [float(h["reading"].get("PM10", 0)) for h in hist_list]
    chart_pm1    = [float(h["reading"].get("PM1",  0)) for h in hist_list]

    # Pass through extra display fields from the ESP32
    extras = {k: float(current[k]) if k in current and k != "VOC" else current.get(k)
              for k in DISPLAY_EXTRAS if k in current}

    # Training distribution warning
    warning = model_warning(chart_pm25) if chart_pm25 else {"level": "none", "message": ""}

    return jsonify({
        "current":    current,
        "prediction": pred,
        "buffer_size": len(hist_list),
        "window_needed": WINDOW_SIZE,
        "aqi_current": aqi_now,
        "aqi_predicted": aqi_pred,
        "extras": extras,
        "warning": warning,
        "training_stats": {
            "pm25_mean": round(TRAIN_STATS.get("features", {}).get("PM25", {}).get("mean", 0), 1),
            "pm25_p5":   round(TRAIN_STATS.get("features", {}).get("PM25", {}).get("p5",  0), 1),
            "pm25_p95":  round(TRAIN_STATS.get("features", {}).get("PM25", {}).get("p95", 0), 1),
        } if TRAIN_STATS else {},
        "chart": {
            "labels":  chart_labels,
            "pm25":    chart_pm25,
            "pm10":    chart_pm10,
            "pm1":     chart_pm1,
        },
        "features_order": FEATURE_ORDER,
    })

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Inject a simulated reading (for demo/testing without real ESP32).
    Pass offset index as JSON: {"index": 42}
    """
    data = request.get_json(force=True, silent=True) or {}
    idx = int(data.get("index", 0))

    # Build a realistic simulated reading
    import math
    base   = 45 + 20 * math.sin(idx * 0.5)
    reading = {
        "PM10": round(base * 1.6 + 5 * math.cos(idx * 0.3), 2),
        "PM25": round(base, 2),
        "PM1":  round(base * 0.55, 2),
        "O3":   round(10 + 5 * math.sin(idx * 0.2), 2),
        "CO":   round(0.8 + 0.3 * math.cos(idx * 0.4), 3),
        "NO":   round(3 + 2 * math.sin(idx * 0.6), 2),
        "NO2":  round(6 + 2 * math.cos(idx * 0.5), 2),
        "NOx":  round(9 + 3 * math.sin(idx * 0.3), 2),
        "CO2":  round(420 + 30 * math.cos(idx * 0.1), 1),
    }

    with lock:
        ts = datetime.now(timezone.utc).isoformat()
        history.append({"timestamp": ts, "reading": reading})
        global latest_reading, latest_prediction
        latest_reading = reading
        pred = run_prediction()
        latest_prediction = pred
        if pred is not None:
            history[-1]["prediction"] = pred

    return jsonify({"status": "simulated", "reading": reading,
                    "prediction": pred, "buffer_size": len(history)})

@app.route("/", methods=["GET"])
def index():
    html_path = os.path.join(BASE_DIR, "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model_weights()
    load_training_stats()
    print("\n" + "=" * 60)
    print("  PM2.5 Prediction Dashboard  –  Server starting")
    print("=" * 60)
    print(f"  Dashboard : http://localhost:5000/")
    print(f"  POST data : POST http://localhost:5000/api/reading")
    print(f"  GET state : GET  http://localhost:5000/api/state")
    print(f"  Simulate  : POST http://localhost:5000/api/simulate")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
