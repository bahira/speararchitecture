"""Iter 5: validate refined-table formulas (beta, blur) + discover missing ones
(Gaussian CDF rational, GELU rational-family replacement).
Run: python refine_fit2.py  -> updates spear_constants.json + results_ops.json
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

import spear_ops as so

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "spear_constants.json")))


def linf(f, ref, lo=-8, hi=8):
    g = so.grid(lo=lo, hi=hi)
    return float(np.max(np.abs(f(g) - ref(g))))


def hill_climb(p, loss_fn, iters=500, scale0=0.15):
    p = np.array(p, float)
    best = loss_fn(p)
    scale = scale0
    rng = np.random.default_rng(11)
    for _ in range(iters):
        q = p * (1.0 + rng.normal(0, scale, size=p.shape))
        l = loss_fn(q)
        if l < best:
            p, best = q, l
            scale *= 0.98
        else:
            scale *= 0.999
    return p, best


# ---- refined-table formulas ----
def beta_new(t):
    return -0.496 * np.cos(3.162 * t) + 0.501


def blur_new(r):
    return 0.999 * np.exp(-0.504 * r * r)


def phi_rat(v, b, c, d):
    """Pure-algebraic Gaussian CDF: 0.5 + 0.5 * (b v)/(c + sqrt(d + v^2))."""
    return 0.5 + 0.5 * (b * v) / (c + np.sqrt(d + v * v))


def gelu_rat(v, a, b, c, d):
    """GELU via the winning SiLU rational family (no exp, no erf)."""
    return v * (a + b * v / (c + np.sqrt(d + v * v)))


def main():
    # ---- 1) beta(t) refined: validity + fidelity vs Nichol-Dhariwal cosine
    t = np.linspace(0, 1, 1001)
    f = lambda u: np.cos((u + 0.008) / 1.008 * np.pi / 2) ** 2
    abar = f(t) / f(0.0)
    beta_cos = 1.0 - abar[1:] / abar[:-1]
    bp = beta_new(t[1:])
    valid = float(np.mean((bp > 0) & (bp < 1)))
    mono = bool(np.all(np.diff(bp) >= -1e-9))
    mdiff = float(np.max(np.abs(bp - beta_cos)))
    print(f"beta_new: valid {valid:.3f} monotone {mono} max|diff vs cosine| {mdiff:.4e}")

    # ---- 2) blur refined: is it the gaussian?
    off = np.arange(-3, 4, dtype=np.float64)
    kg = np.exp(-off ** 2 / 2.0)
    kg /= kg.sum()
    kn = blur_new(off)
    kn /= kn.sum()
    print(f"blur_new: kernel max|diff| vs gaussian(sigma=1) {np.max(np.abs(kg - kn)):.4e}")

    # ---- 3) Gaussian CDF: fill the missing 'refined rational form'
    g = so.grid(lo=-6, hi=6)
    r = least_squares(lambda p: (phi_rat(g, *p) - so.sigmoid_ref(g)),
                      [1.0, 0.6, 1.8], bounds=([0.5, 0.01, 0.1], [2.0, 4.0, 8.0]))
    loss = lambda p: linf(lambda v: phi_rat(v, *p), so.sigmoid_ref, -6, 6)
    pp, ep = hill_climb(r.x, loss)
    print(f"phi rational: Linf {ep:.3e}  params={np.round(pp, 6).tolist()}")

    # ---- 4) GELU: rational family (replaces failed linear-cap)
    g8 = so.grid(lo=-8, hi=8)
    r2 = least_squares(lambda p: (gelu_rat(g8, *p) - so.gelu_ref(g8)),
                       [0.02, 0.65, 0.05, 3.2],
                       bounds=([-0.5, 0.01, 0.001, 0.1], [1.0, 2.0, 2.0, 16.0]))
    loss2 = lambda p: linf(lambda v: gelu_rat(v, *p), so.gelu_ref)
    pg, eg = hill_climb(r2.x, loss2)
    print(f"gelu rational: Linf {eg:.3e} (old linear-cap {C.get('gelu_linf_old', 6.9e-2):.1e}) "
          f"params={np.round(pg, 6).tolist()}")

    # ---- persist
    C["gelu_linf_old"] = C.get("refit2", {}).get("gelu_linf", 6.9e-2)
    C["phi"] = [float(v) for v in pp]
    C["gelu2"] = [float(v) for v in pg]
    C["validated"] = dict(beta_new_valid=valid, beta_new_monotone=mono,
                          beta_new_maxdiff=mdiff, blur_new_kernel_maxdiff=float(np.max(np.abs(kg - kn))))
    with open(os.path.join(HERE, "spear_constants.json"), "w") as fh:
        json.dump(C, fh, indent=2)

    rp = os.path.join(HERE, "results_ops.json")
    res = json.load(open(rp))
    res["refit3"] = dict(beta=dict(valid=valid, monotone=mono, maxdiff=mdiff),
                         blur_kernel_maxdiff=float(np.max(np.abs(kg - kn))),
                         phi_linf=ep, gelu_rational_linf=eg)
    with open(rp, "w") as fh:
        json.dump(res, fh, indent=2)
    print("updated spear_constants.json + results_ops.json")


if __name__ == "__main__":
    main()
