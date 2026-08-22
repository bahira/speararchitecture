"""Drop-in algebraic activations (torch + numpy), constants fitted by
grounded-loop minimax refit (see README.md / spear_constants.json).

Usage:
    import spear_activations as sa
    y = sa.silu(x_torch)              # fonctions tensorielles
    m = sa.SpearSiLU()                # nn.Module drop-in pour nn.Sequential
    w = sa.exact_wasserstein1(p, q)   # W1 exact O(n)

Self-check (parite vs references exactes) : python spear_activations.py
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))

C = {
    "silu": [0.50000000000002930, 0.51644495786218860, 0.01000000000000129,
             2.85377300701158100],
    "gelu2": [0.49999999999990750, 0.50558992755953450, 0.00100000000000000,
              0.68219927331460940],
    "softplus": [9.17213789289023200, 9.18217225458581300, 5.48952043564750100],
    "phi": [1.06124354675369250, 0.00976429170117038, 3.85682885374298620],
    "tanh_indomain": [0.06381872026634902, 1.00003711628576530,
                      0.39676795752521077],
}
_p = os.path.join(HERE, "spear_constants.json")
if os.path.exists(_p):
    with open(_p) as f:
        C.update({k: v for k, v in json.load(f).items() if k in C})


# ---------------- torch ----------------
def silu(t):
    a, b, c, d = C["silu"]
    return t * (a + b * t / (c + torch.sqrt(d + t * t)))


def gelu(t):
    a, b, c, d = C["gelu2"]
    return t * (a + b * t / (c + torch.sqrt(d + t * t)))


def softplus(t):
    sa, sb, sc = C["softplus"]
    u = torch.exp(-torch.abs(t))
    return torch.clamp(t, min=0.0) + u * (sa + u) / (sb + sc * u)


def phi(t):
    pb, pc, pd = C["phi"]
    t2 = t * t
    return torch.clamp(0.5 + 0.5 * (pb * t) / (pc + torch.sqrt(pd + t2)),
                       min=0.0, max=1.0)


def softsign(t):
    """Gate borné (-1,1) EXACT, 6 unités ALU, zéro transcendance (HoF 'sigmoid fast').
    Honnête : c'est softsign exact — une primitive de gating, pas un sigmoid."""
    return t / (1.0 + torch.abs(t))


class SpearSoftsign(nn.Module):
    def forward(self, x):
        return softsign(x)


class SpearSiLU(nn.Module):
    def forward(self, x):
        return silu(x)


class SpearGELU(nn.Module):
    def forward(self, x):
        return gelu(x)


class SpearSoftplus(nn.Module):
    def forward(self, x):
        return softplus(x)


# ---------------- numpy ----------------
def _np_rstep(x, a, b, c, d):
    x2 = x * x
    return x * (a + b * x / (c + np.sqrt(d + x2)))


def silu_np(x):
    return _np_rstep(x, *C["silu"])


def gelu_np(x):
    return _np_rstep(x, *C["gelu2"])


def phi_np(x):
    pb, pc, pd = C["phi"]
    return np.clip(0.5 + 0.5 * (pb * x) / (pc + np.sqrt(pd + x * x)), 0.0, 1.0)


def softplus_np(x):
    sa, sb, sc = C["softplus"]
    u = np.exp(-np.abs(x))
    return np.maximum(x, 0.0) + u * (sa + u) / (sb + sc * u)


def tanh_indomain_np(x):
    ta, tb, tc = C["tanh_indomain"]
    v = (x + ta * x ** 3) / (tb + tc * x ** 2)
    return np.clip(v, -1.0, 1.0)


def exact_wasserstein1(p, q):
    """W1 entre histogrammes normalises : O(n), une passe, en unites de bins."""
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


# ---------------- self-check ----------------
if __name__ == "__main__":
    from scipy.special import erf

    g = np.linspace(-8.0, 8.0, 200_001)
    g4 = np.linspace(-4.0, 4.0, 200_001)
    checks = [
        ("silu", silu_np(g), g / (1 + np.exp(-g)), 4.5e-2),
        ("gelu", gelu_np(g), 0.5 * g * (1 + erf(g / np.sqrt(2))), 5.5e-2),
        ("softplus", softplus_np(g), np.logaddexp(0, g), 2.5e-4),
        ("phi[-4,4]", phi_np(g4), 0.5 * (1 + erf(g4 / np.sqrt(2))), 5.0e-3),
        ("phi[-8,8]", phi_np(g), 0.5 * (1 + erf(g / np.sqrt(2))), 5.0e-2),
    ]
    gt = np.linspace(-1.0, 1.0, 100_001)
    checks.append(("tanh[-1,1]", tanh_indomain_np(gt), np.tanh(gt), 2.0e-5))

    ok = True
    print(f"{'op':12s} {'Linf':>10s} {'seuil':>10s}")
    for name, got, ref, thr in checks:
        err = float(np.max(np.abs(got - ref)))
        status = "PASS" if err <= thr else "FAIL"
        ok &= err <= thr
        print(f"{name:12s} {err:10.3e} {thr:10.1e}  {status}")

    rng = np.random.default_rng(0)
    p = rng.random(64); p /= p.sum()
    q = rng.random(64); q /= q.sum()
    ours = exact_wasserstein1(p, q)
    cdf = float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())
    print(f"w1 identity  {abs(ours - cdf):.2e}")
    sys.exit(0 if ok else 1)
