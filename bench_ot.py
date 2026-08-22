"""BT11: exact Wasserstein-1 O(D) 1-pass vs iterative solvers.
Baselines: scipy wasserstein_distance (production ref) + log-domain Sinkhorn (entropic, stable).
Run: python bench_ot.py
"""
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

rng = np.random.default_rng(3)


def exact_w1(p, q):
    """SPEAR closed form: W1 = sum |CDF_p - CDF_q| (bin units), single pass."""
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def sinkhorn_log_w1(p, q, iters=3000, eps=0.02):
    """Log-domain stabilized entropic OT on grid metric |i-j| (bin units)."""
    n = len(p)
    C = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    f = np.zeros(n)
    g = np.zeros(n)
    lp, lq = np.log(p + 1e-300), np.log(q + 1e-300)
    for _ in range(iters):
        f = eps * lp - eps * logsumexp((g[None, :] - C) / eps, axis=1)
        g = eps * lq - eps * logsumexp((f[:, None] - C) / eps, axis=0)
    P = np.exp((f[:, None] + g[None, :] - C) / eps)
    return float((P * C).sum())


def main():
    D = 256
    p = rng.random(D)
    p /= p.sum()
    q = rng.random(D)
    q /= q.sum()

    t0 = time.perf_counter()
    w1 = exact_w1(p, q)
    t_ex = time.perf_counter() - t0

    t0 = time.perf_counter()
    ws_ref = wasserstein_distance(np.arange(D), np.arange(D), p, q)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    ws_sk = sinkhorn_log_w1(p, q)
    t_sk = time.perf_counter() - t0

    print(f"D={D}   exact W1 = {w1:.6f} bins")
    print(f"SPEAR exact  : {t_ex*1e3:8.3f} ms")
    print(f"scipy ref    : {ws_ref:.6f}   {t_ref*1e3:8.3f} ms  relerr {abs(ws_ref-w1)/w1:.1e}")
    print(f"sinkhorn-log : {ws_sk:.6f}   {t_sk*1e3:8.3f} ms  relerr {abs(ws_sk-w1)/w1:.1e}  (3000 it)")
    print(f"SPEEDUP vs scipy      : x{t_ref/t_ex:.0f}")
    print(f"SPEEDUP vs sinkhorn   : x{t_sk/t_ex:.0f}")


if __name__ == "__main__":
    main()
