"""Audit du spear-hall-of-fame : chaque forme candidate est implémentée
verbatim et confrontée au ground truth sur son domaine nominal.
Verdict par entrée : ADOPTÉ / SLOT RAPIDE / REJETÉ.

Run: python hof_verify.py
"""
import json

import numpy as np
from scipy.special import erf

import spear_ops as so


def linf(f, lo, hi, n=200001):
    g = np.linspace(lo, hi, n)
    return float(np.max(np.abs(f(g) - REF[g >= -1e18] if False else f(g))))  # placeholder


def err_vs(ref, f, lo=-8.0, hi=8.0, n=200001):
    g = np.linspace(lo, hi, n)
    return float(np.max(np.abs(f(g) - ref(g))))


# ---------- références ----------
sig_ref = lambda v: 1.0 / (1.0 + np.exp(-v))
silu_ref = lambda v: v * sig_ref(v)
phi_true = lambda v: 0.5 * (1 + erf(v / np.sqrt(2)))
sp_true = lambda v: np.logaddexp(0.0, v)
tanh_true = np.tanh

# ---------- candidats Hall of Fame (verbatim) ----------
def silu_hof(v):  # seed 910001, metric publié 7.8e-4
    r = v / (0.815 + np.sqrt(np.abs(0.909361 + v * v)))
    return 0.997587 * (v * (0.501 + 0.587 * r)) - 0.005749


def gelu_hof(v):  # seed 950303, metric publié 5.3e-4
    return 0.997729 * (v * np.minimum(1.002, np.maximum(0.306923 * v + 0.501, 0))) - 0.004004


def phi_hof(v):  # seed 990303, metric publié 1.4e-4, 9 unités, zéro erf
    return 0.543518 * (v / np.maximum(1.559472, 0.337459 + np.abs(v))) + 0.5


def phi_hof_fast(v):  # 6 unités
    return v / np.maximum(1.531472, np.abs(v))


def softplus_hof(v):  # seed 920303, metric publié 2.3e-4, 9 unités, ZÉRO exp
    a = np.minimum(v, 9.67777) * (v + 11.264046)
    b = v * np.maximum(v + 6.712061, -678.942377)
    return 0.05564874688602696 * np.maximum(a, b) + 0.6507988992446906


def softplus_hof_fast(v):  # 2 unités !
    return (v + 5.281056) * v


def softsign_gate(v):  # 'sigmoid fast' : 6 unités, zéro transcendance
    return v / (1.0 + np.abs(v))


def tanh_hof(v):  # rl_distillation seed 777, constantes raffinées
    return 1.143472 * ((v + 0.135 * v ** 3) / (0.584 + 0.75 * v ** 2))


# ---------- champions actuels du repo (spear_constants.json) ----------
C = json.load(open("spear_constants.json"))


def phi_ours(v):
    b, c, d = C["phi"]
    return np.clip(0.5 + 0.5 * (b * v) / (c + np.sqrt(d + v * v)), 0.0, 1.0)


def silu_ours(v):
    a, b, c, d = C["silu"]
    return v * (a + b * v / (c + np.sqrt(d + v * v)))


def sp_ours(v):
    sa, sb, sc = C["softplus"]
    u = np.exp(-np.abs(v))
    return np.maximum(v, 0) + u * (sa + u) / (sb + sc * u)


def tanh_ours(v):
    ta, tb, tc = C["tanh_indomain"]
    return np.clip((v + ta * v ** 3) / (tb + tc * v ** 2), -1, 1)


def main():
    print(f"{'entrée':22s} {'domaine':10s} {'err HoF':>10s} {'err nôtre':>10s} {'claim':>10s}")
    rows = []

    e = err_vs(silu_ref, silu_hof); e0 = err_vs(silu_ref, silu_ours)
    rows.append(("silu affine-corrigée", "[-8,8]", e, e0, 7.8e-4))
    e = err_vs(lambda v: 0.5 * v * (1 + erf(v / np.sqrt(2))), gelu_hof)
    rows.append(("gelu lin-cap corrigée", "[-8,8]", e, None, 5.3e-4))
    e = err_vs(phi_true, phi_hof); e0 = err_vs(phi_true, phi_ours)
    rows.append(("phi x/max() 9u", "[-8,8]", e, e0, 1.4e-4))
    e4 = err_vs(phi_true, phi_hof, -4, 4)
    rows.append(("phi x/max() 9u", "[-4,4]", e4, None, 1.4e-4))
    e = err_vs(sp_true, softplus_hof); e0 = err_vs(sp_true, sp_ours)
    rows.append(("softplus min/max SANS exp", "[-8,8]", e, e0, 2.3e-4))
    e = err_vs(sp_true, softplus_hof_fast, -6, 6)
    rows.append(("softplus fast (x+a)x 2u", "[-6,6]", e, None, 3.2e-3))
    e = err_vs(sig_ref, lambda v: (softsign_gate(v) + 1) / 2, -8, 8)
    rows.append(("gate softsign->sigmoid", "[-8,8]", e, None, 7.2e-4))
    e = err_vs(tanh_true, tanh_hof, -1, 1); e0 = err_vs(tanh_true, tanh_ours, -1, 1)
    rows.append(("tanh Padé raffinée", "[-1,1]", e, e0, 1.6e-4))
    e = err_vs(tanh_true, tanh_hof, -3, 3)
    rows.append(("tanh Padé raffinée", "[-3,3]", e, None, "-"))

    for name, dom, ehof, eours, claim in rows:
        co = f"{eours:.2e}" if eours is not None else "     -"
        cc = f"{claim:.1e}" if isinstance(claim, float) else str(claim)
        print(f"{name:22s} {dom:10s} {ehof:10.2e} {co:>10s} {cc:>10s}")

    # sauvegarde brute pour décision
    out = {
        "silu_hof": err_vs(silu_ref, silu_hof),
        "gelu_hof": err_vs(lambda v: 0.5 * v * (1 + erf(v / np.sqrt(2))), gelu_hof),
        "phi_hof_global": err_vs(phi_true, phi_hof),
        "phi_hof_dom4": e4,
        "phi_ours_global": err_vs(phi_true, phi_ours),
        "softplus_hof": err_vs(sp_true, softplus_hof),
        "softplus_hof_fast": err_vs(sp_true, softplus_hof_fast, -6, 6),
        "softplus_ours": err_vs(sp_true, sp_ours),
        "tanh_hof_dom1": err_vs(tanh_true, tanh_hof, -1, 1),
        "tanh_ours_dom1": err_vs(tanh_true, tanh_ours, -1, 1),
    }
    json.dump(out, open("hof_results.json", "w"), indent=2)
    print("\nsaved hof_results.json")


if __name__ == "__main__":
    main()
