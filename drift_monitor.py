"""Live drift monitor powered by exact Wasserstein-1 O(D).

Simulated feature stream: clean -> progressive mean drift -> variance drift.
Threshold calibrated online on the first clean window (mean+6*sigma), frozen after.
Compares wall-clock of our O(n) W1 against scipy per tick.

Run: python drift_monitor.py
"""
import time

import numpy as np
from scipy.stats import wasserstein_distance  # timing reference only

rng = np.random.default_rng(7)
BINS, LO, HI = 96, -5.0, 5.0
edges = np.linspace(LO, HI, BINS + 1)

ref_p, _ = np.histogram(rng.standard_normal(100_000), bins=edges)
ref_p = ref_p / ref_p.sum()

CALIB, DRIFT_AT, TICKS, BATCH = 25, 40, 120, 256
calib, thr, detected_at, false_alarms = [], None, None, 0
n_drift_alerts = 0
t_exact = t_scipy = 0.0

print(f"{'tick':>4s} {'W1(bins)':>9s} {'thr':>7s}  status")
for t in range(TICKS):
    mu = 0.0 if t < DRIFT_AT else min(1.5, (t - DRIFT_AT) * 0.075)
    sd = 1.0 if t < DRIFT_AT + 40 else min(1.8, 1.0 + (t - DRIFT_AT - 40) * 0.04)
    x = rng.standard_normal(BATCH) * sd + mu
    h, _ = np.histogram(x, bins=edges)
    q = h / h.sum()

    t0 = time.perf_counter()
    w1 = float(np.abs(np.cumsum(ref_p) - np.cumsum(q)).sum())
    t_exact += time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = wasserstein_distance(np.arange(BINS), np.arange(BINS), ref_p, q)
    t_scipy += time.perf_counter() - t0

    if t < CALIB:
        calib.append(w1)
        status, thr_s = "calib", "      -"
    else:
        if thr is None:
            arr = np.array(calib)
            thr = max(float(arr.mean() + 6.0 * arr.std()), 1e-4)
            print(f"-- threshold calibrated on clean window: {thr:.4f} bins")
        if w1 <= thr:
            status = "ok"
        elif t >= DRIFT_AT:
            status = "DRIFT"
            n_drift_alerts += 1
            if detected_at is None:
                detected_at = t
        else:
            status = "FALSE?"
            false_alarms += 1
        thr_s = f"{thr:7.4f}"

    print(f"{t:4d} {w1:9.4f} {thr_s}  {status}")

print("\n=== summary ===")
latency = None if detected_at is None else detected_at - DRIFT_AT
print(f"mean drift injected at tick {DRIFT_AT} -> detected at tick {detected_at} "
      f"(latency {latency} ticks = {None if latency is None else latency*BATCH} samples)")
print(f"variance drift from tick {DRIFT_AT+40}: {n_drift_alerts}/{TICKS-DRIFT_AT} post-drift ticks in DRIFT")
print(f"false alarms during clean phase: {false_alarms}")
print(f"W1 exact cumulative: {t_exact*1e3:.1f} ms | scipy: {t_scipy*1e3:.1f} ms "
      f"| speedup x{t_scipy/max(t_exact,1e-12):.1f}")
