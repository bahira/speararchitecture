"""Iter 2: richer/minimax refits + torch.compile fusion benchmark.
Run: python refine_fit.py  (updates spear_constants.json, appends results_ops.json)
"""
import json
import os
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

import spear_ops as so

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "spear_constants.json")))


def linf(f, ref, lo=-8, hi=8):
    g = so.grid(lo=lo, hi=hi)
    return float(np.max(np.abs(f(g) - ref(g))))


def hill_climb(p, loss_fn, iters=500, scale0=0.15):
    """Minimax polish: 500 random perturbations, keep improvements, decay scale."""
    p = np.array(p, float)
    best = loss_fn(p)
    scale = scale0
    rng = np.random.default_rng(7)
    for _ in range(iters):
        q = p * (1.0 + rng.normal(0, scale, size=p.shape))
        l = loss_fn(q)
        if l < best:
            p, best = q, l
            scale *= 0.98
        else:
            scale *= 0.999
    return p, best


def silu5(v, a, b, c, d):
    return v * (a + b * v / (c + np.sqrt(d + v * v)))


def gelu_q(v, a, b, c, k):
    return v * np.minimum(k, np.maximum((c * v + b) * v + a, 0.0))


def main():
    g = so.grid(lo=-8, hi=8)

    # ---- SiLU: 4-param family + minimax polish
    def loss_silu(p):
        return float(np.max(np.abs(silu5(g, *p) - so.silu_ref(g))))

    r = least_squares(lambda p: (silu5(g, *p) - so.silu_ref(g)), [*C["silu"], 1.0],
                      bounds=([0, 0, 0.01, 0.01], [2, 2, 4, 4]))
    p4, e4 = hill_climb(r.x, loss_silu)
    print(f"silu 4-param: Linf {linf(lambda v: silu5(v, *r.x), so.silu_ref):.3e} "
          f"-> minimax {e4:.3e}  params={np.round(p4, 6).tolist()}")

    # ---- GELU: quadratic-shoulder piecewise + minimax polish
    def loss_gelu(p):
        return float(np.max(np.abs(gelu_q(g, *p) - so.gelu_ref(g))))

    r2 = least_squares(lambda p: gelu_q(g, *p) - so.gelu_ref(g),
                       [0.02, 0.55, 0.45, 1.0],
                       bounds=([-0.5, 0, -2, 0.5], [0.5, 2, 2, 2]))
    pg, eg = hill_climb(r2.x, loss_gelu)
    print(f"gelu quad-shoulder: Linf {linf(lambda v: gelu_q(v, *r2.x), so.gelu_ref):.3e} "
          f"-> minimax {eg:.3e}  params={np.round(pg, 6).tolist()}")

    # ---- softplus minimax polish on existing family
    def loss_sp(p):
        return linf(lambda v: so.softplus_pub(v, *p), so.softplus_ref, -6, 6)

    ps, es = hill_climb(C["softplus"], loss_sp, scale0=0.1)
    print(f"softplus minimax: {es:.3e}  params={np.round(ps, 6).tolist()}")

    # ---- torch.compile fusion benchmark
    comp_rows = []
    try:
        import torch
        import torch.nn.functional as F
        a4, b4, c4, d4 = [float(v) for v in p4]
        ga_, gb_, gc_, gk_ = [float(v) for v in pg]

        def spear_silu_t(v):
            return v * (a4 + b4 * v / (c4 + torch.sqrt(d4 + v * v)))

        def spear_gelu_t(v):
            return v * torch.clamp(torch.clamp((gc_ * v + gb_) * v + ga_, min=0.0), max=gk_)

        cs, cg = torch.compile(spear_silu_t), torch.compile(spear_gelu_t)
        x = torch.randn(4_000_000)
        for name, ref, spear, compiled in [
                ("silu", F.silu, spear_silu_t, cs), ("gelu", F.gelu, spear_gelu_t, cg)]:
            t0 = time.perf_counter()
            for _ in range(3):
                compiled(x)
            tcomp = time.perf_counter() - t0
            for f in (ref, spear, compiled):
                for _ in range(3):
                    f(x)
            t0 = time.perf_counter()
            for _ in range(30):
                ref(x)
            tr = (time.perf_counter() - t0) / 30
            t0 = time.perf_counter()
            for _ in range(30):
                spear(x)
            ts = (time.perf_counter() - t0) / 30
            t0 = time.perf_counter()
            for _ in range(30):
                compiled(x)
            tc = (time.perf_counter() - t0) / 30
            comp_rows.append(dict(op=name, ref_ms=tr * 1e3, eager_ms=ts * 1e3,
                                  compiled_ms=tc * 1e3, compile_overhead_s=tcomp))
            print(f"{name}: ref {tr*1e3:.2f}ms | eager {ts*1e3:.2f}ms ({tr/ts:.2f}x) | "
                  f"compiled {tc*1e3:.2f}ms ({tr/tc:.2f}x)")
    except Exception as e:
        print("torch.compile failed:", repr(e)[:200])

    # ---- persist
    C["silu"] = [float(v) for v in p4]
    C["gelu"] = [float(v) for v in pg]
    C["softplus"] = [float(v) for v in ps]
    C["family"] = {"silu": "v*(a+b*v/(c+sqrt(d+v*v)))",
                   "gelu": "v*clamp(clamp((c*v+b)*v+a,0),k)",
                   "softplus": "max(v,0)+u*(A+u)/(B+C*u), u=e^-|v|"}
    with open(os.path.join(HERE, "spear_constants.json"), "w") as f:
        json.dump(C, f, indent=2)

    rp = os.path.join(HERE, "results_ops.json")
    res = json.load(open(rp))
    res["refit2"] = dict(silu_linf=e4, gelu_linf=eg, softplus_linf=es,
                         torch_compile=comp_rows)
    with open(rp, "w") as f:
        json.dump(res, f, indent=2)
    print("updated spear_constants.json + results_ops.json")


if __name__ == "__main__":
    main()
