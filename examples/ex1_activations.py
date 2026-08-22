"""Exemple 1 - activations algebriques : drop-in PyTorch + parite d'entrainement.

Montre que remplacer nn.SiLU()/nn.GELU() par les versions SPEAR ne degrade pas
l'entrainement (parite verifiee sur tache synthetique), et mesure le cout elementaire.

Run: python ex1_activations.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import spear_activations as sa


def make_net(act_factory):
    return nn.Sequential(
        nn.Linear(1, 64), act_factory(),
        nn.Linear(64, 64), act_factory(),
        nn.Linear(64, 1),
    )


def train(act_factory, steps=400):
    torch.manual_seed(0)
    net = make_net(act_factory)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(net(X), Y)
        loss.backward()
        opt.step()
    return float(loss.detach())


# --- 1) parite : meme reseau, activation standard vs algebrique -------------
torch.manual_seed(0)
X = torch.rand(4096, 1) * 12 - 6
Y = torch.sin(X) + 0.05 * torch.randn_like(X)

mse_ref = train(nn.SiLU)
mse_spear = train(sa.SpearSiLU)
print(f"MSE final   F.silu        : {mse_ref:.5f}")
print(f"MSE final   SpearSiLU     : {mse_spear:.5f}  "
      f"(ecart rel {abs(mse_spear - mse_ref) / mse_ref * 100:.1f}%)")

# --- 2) cout elementaire (eager CPU : ATen fusé gagne souvent, cf README) ---
x = torch.randn(4_000_000)
for name, fn in [("F.silu", F.silu), ("spear_silu", sa.silu)]:
    for _ in range(3):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(20):
        fn(x)
    print(f"{name:12s} {(time.perf_counter() - t0) / 20 * 1e9 / x.numel():6.1f} ns/elem")

# --- 3) CDF gaussienne sans erf --------------------------------------------
print("\nCDF gaussienne purement algebrique (pas de erf) :")
for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
    got = float(sa.phi(torch.tensor(v)))
    ref = 0.5 * (1 + math.erf(v / math.sqrt(2)))
    print(f"  phi({v:+.0f}) spear={got:.5f}  exact={ref:.5f}  diff={abs(got-ref):.1e}")
