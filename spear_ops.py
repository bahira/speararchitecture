"""SPEAR algebraic operators: falsification + refit + speed benchmark.
Run: python spear_ops.py
Outputs: spear_constants.json (refit constants), results_ops.json, printed tables.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))


def grid(n=20001, lo=-8.0, hi=8.0):
    return np.linspace(lo, hi, n)


# ---------------- references (ground truth) ----------------
def sigmoid_ref(v):
    return 1.0 / (1.0 + np.exp(-v))


def silu_ref(v):
    return v * sigmoid_ref(v)


def gelu_ref(v):
    return 0.5 * v * (1.0 + erf(v / np.sqrt(2.0)))


def softplus_ref(v):
    return np.logaddexp(0.0, v)


# ---------------- paper formulas (verbatim) ----------------
def silu_pub(v, a=0.501, b=0.589, c=0.83):
    return v * (a + b * v / (c + np.sqrt(1.0 + v * v)))


def gelu_pub(v, a=0.308, b=0.501, k=1.002):
    return v * np.minimum(k, np.maximum(a * v + b, 0.0))


def tanh_pub(v):
    return (v + 0.145 * v ** 3) / (0.556 + 0.75 * v ** 2)


def tanh_family(v, a, b, c):
    return (v + a * v ** 3) / (b + c * v ** 2)


def softplus_pub(v, A=6.0, B=6.0, C=4.0):
    # paper gives no constants; structure = max(x,0) + Padé[2/2] of log1p(e^-|x|)
    u = np.exp(-np.abs(v))
    return np.maximum(v, 0.0) + u * (A + u) / (B + C * u)


def blur_pub(r):
    return 0.427 * np.exp(np.cos(r)) - 0.14


def beta_pub(t, c=1.0):
    return -5.61 * np.exp(np.cos(np.minimum(c, t) - t * t)) + 3.05


def maxerr(f, ref, lo=-8.0, hi=8.0):
    g = grid(lo=lo, hi=hi)
    return float(np.max(np.abs(f(g) - ref(g))))


# ---------------- refit (same functional family, new constants) ----------------
def refit():
    out = {}
    g = grid(lo=-8, hi=8)
    r = least_squares(lambda p: silu_pub(g, *p) - silu_ref(g), [0.501, 0.589, 0.83])
    out["silu"] = [float(v) for v in r.x]

    best, bp = None, None
    for s in ([0.308, 0.501, 1.002], [0.5, 0.3, 1.0], [0.2, 0.6, 1.2]):
        r = least_squares(lambda p: gelu_pub(g, *p) - gelu_ref(g), s,
                          bounds=([0.0, 0.0, 0.5], [2.0, 2.0, 2.0]))
        e = maxerr(lambda v, p=r.x: gelu_pub(v, *p), gelu_ref)
        if best is None or e < best:
            best, bp = e, r.x
    out["gelu"] = [float(v) for v in bp]

    g0 = grid(lo=-6, hi=6)
    r = least_squares(lambda p: softplus_pub(g0, *p) - softplus_ref(g0), [6, 6, 4],
                      bounds=([1.0, 1.0, 1.0], [20.0, 20.0, 20.0]))
    out["softplus"] = [float(v) for v in r.x]

    gt = grid(lo=-1, hi=1)
    r = least_squares(lambda p: tanh_family(gt, *p) - np.tanh(gt), [0.145, 0.556, 0.75],
                      bounds=([0.0, 0.5, 0.0], [1.0, 2.0, 3.0]))
    out["tanh_indomain"] = [float(v) for v in r.x]

    with open(os.path.join(HERE, "spear_constants.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------- multimodal: image blur kernel + diffusion beta ----------------
def image_bench():
    from scipy import ndimage
    rng = np.random.default_rng(1)
    img = ndimage.gaussian_filter(rng.standard_normal((256, 256)), 4.0)
    img = (img - img.min()) / np.ptp(img)
    off = np.arange(-3, 4, dtype=np.float64)
    kg = np.exp(-off ** 2 / 2.0)
    kg /= kg.sum()
    kp = np.clip(blur_pub(off), 0.0, None)
    kp /= kp.sum()

    def blur(im, k):
        return ndimage.correlate1d(ndimage.correlate1d(im, k, 0), k, 1)

    t0 = time.perf_counter()
    for _ in range(30):
        bg = blur(img, kg)
    tg = (time.perf_counter() - t0) / 30
    t0 = time.perf_counter()
    for _ in range(30):
        bpub = blur(img, kp)
    tp = (time.perf_counter() - t0) / 30
    mse = float(np.mean((bg - bpub) ** 2))
    return dict(psnr_vs_gaussian=10 * np.log10(1.0 / mse),
                t_gauss_us=tg * 1e6, t_pub_us=tp * 1e6,
                kernel_max_abs_diff=float(np.max(np.abs(kg - kp))))


def beta_check():
    t = np.linspace(0.0, 1.0, 1001)
    f = lambda u: np.cos((u + 0.008) / 1.008 * np.pi / 2) ** 2
    abar = f(t) / f(0.0)
    beta_cos = np.clip(1.0 - abar[1:] / abar[:-1], 1e-5, 0.999)
    bp = beta_pub(t, 1.0)[1:]
    valid = float(np.mean((bp > 0.0) & (bp < 1.0)))
    bpc = np.clip(bp, 1e-5, 0.999)
    return dict(frac_valid_beta=valid,
                max_abs_diff_vs_cosine_clamped=float(np.max(np.abs(bpc - beta_cos))))


# ---------------- torch speed benchmark ----------------
def torch_bench(C):
    import torch
    import torch.nn.functional as F
    a, b, c = C["silu"]
    ga, gb, gk = C["gelu"]
    sa, sb, sc = C["softplus"]

    def spear_silu(v):
        return v * (a + b * v / (c + torch.sqrt(1.0 + v * v)))

    def spear_gelu(v):
        return v * torch.clamp(ga * v + gb, min=0.0, max=gk)

    def spear_sigmoid(v):
        return 1.0 - 1.0 / (1.0 + torch.exp(-v))

    def spear_softplus(v):
        u = torch.exp(-torch.abs(v))
        return torch.clamp(v, min=0.0) + u * (sa + u) / (sb + sc * u)

    def spear_tanh(v):
        ta, tb, tc = C["tanh_indomain"]
        return (v + ta * v ** 3) / (tb + tc * v ** 2)

    N = 4_000_000
    x = torch.randn(N)
    fns = [
        ("F.silu", F.silu, spear_silu),
        ("F.gelu(erf)", F.gelu, spear_gelu),
        ("torch.sigmoid", torch.sigmoid, spear_sigmoid),
        ("F.softplus", F.softplus, spear_softplus),
        ("torch.tanh", torch.tanh, spear_tanh),
    ]
    rows = []
    for name, ref, spear in fns:
        for f in (ref, spear):
            for _ in range(5):
                f(x)
        t0 = time.perf_counter()
        for _ in range(30):
            ref(x)
        tr = (time.perf_counter() - t0) / 30
        t0 = time.perf_counter()
        for _ in range(30):
            spear(x)
        ts = (time.perf_counter() - t0) / 30
        rows.append(dict(op=name, ref_ms=tr * 1e3, spear_ms=ts * 1e3,
                         speedup=tr / ts, ns_per_elem_ref=tr / N * 1e9,
                         ns_per_elem_spear=ts / N * 1e9))
    return rows


def main():
    print("=== SPEAR ops: falsification + refit ===", flush=True)
    C = refit()

    rows = []
    rows.append(("silu", maxerr(silu_pub, silu_ref),
                 maxerr(lambda v: silu_pub(v, *C["silu"]), silu_ref)))
    rows.append(("gelu", maxerr(gelu_pub, gelu_ref),
                 maxerr(lambda v: gelu_pub(v, *C["gelu"]), gelu_ref)))
    rows.append(("tanh [-8,8]", maxerr(tanh_pub, np.tanh), None))
    rows.append(("tanh [-1,1] pub", maxerr(tanh_pub, np.tanh, -1, 1),
                 maxerr(lambda v: tanh_family(v, *C["tanh_indomain"]), np.tanh, -1, 1)))
    rows.append(("softplus", maxerr(softplus_pub, softplus_ref),
                 maxerr(lambda v: softplus_pub(v, *C["softplus"]), softplus_ref)))
    rows.append(("sigmoid (1-1/(1+e^-x))", maxerr(sigmoid_ref, sigmoid_ref), None))

    print(f"{'op':24s} {'pub_err':>10s} {'refit_err':>10s}")
    for n, e1, e2 in rows:
        print(f"{n:24s} {e1:10.3e} " + (f"{e2:10.3e}" if e2 is not None else "         -"))

    ib = image_bench()
    bc = beta_check()
    print("\n=== image kernel (blur, in-domain |x|<=3) ===")
    print(f"PSNR vs gaussian sigma=1: {ib['psnr_vs_gaussian']:.2f} dB | kernel max diff {ib['kernel_max_abs_diff']:.4f}")
    print(f"time gauss {ib['t_gauss_us']:.0f} us | pub {ib['t_pub_us']:.0f} us")

    print("\n=== diffusion beta(t) validity ===")
    print(f"frac of beta in (0,1): {bc['frac_valid_beta']:.3f} | max |diff| vs cosine (clamped): {bc['max_abs_diff_vs_cosine_clamped']:.4f}")

    print("\n=== torch CPU speed (4M elems, 30 reps) ===")
    tb = torch_bench(C)
    print(f"{'op':18s} {'ref ms':>8s} {'spear ms':>9s} {'speedup':>8s}")
    for r in tb:
        print(f"{r['op']:18s} {r['ref_ms']:8.2f} {r['spear_ms']:9.2f} {r['speedup']:7.2f}x")

    with open(os.path.join(HERE, "results_ops.json"), "w") as f:
        json.dump(dict(parity=[(n, e1, e2) for n, e1, e2 in rows],
                       constants=C, image=ib, beta=bc, torch_speed=tb), f, indent=2)
    print("\nsaved: spear_constants.json, results_ops.json")


if __name__ == "__main__":
    main()
