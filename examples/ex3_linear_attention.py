"""Exemple 3 - attention lineaire O(N) pour contextes longs.

Mesure le crossover softmax/lineaire sur VOTRE machine et montre l'integration
dans un bloc transformer standard.

Run: python ex3_linear_attention.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spear_llm import LinearAttn, SoftmaxAttn

torch.set_num_threads(4)
D, H = 64, 4   # hd=16 -> crossover theorique ~hd^2=256 tokens


def fwd_bwd_ms(model, B, T, reps=5):
    x = torch.randn(B, T, D)
    for _ in range(2):
        model(x).sum().backward()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x).sum().backward()
    return (time.perf_counter() - t0) / reps * 1e3


print(f"{'T':>6s} {'softmax ms':>11s} {'lineaire ms':>12s} {'speedup':>8s}")
for T in (512, 2048, 8192):
    sm = fwd_bwd_ms(SoftmaxAttn(D, H), 1, T)
    ln = fwd_bwd_ms(LinearAttn(D, H), 1, T)
    print(f"{T:6d} {sm:11.0f} {ln:12.0f} {sm / ln:7.2f}x")

print("""
Integration dans un bloc standard :

    class Bloc(nn.Module):
        def __init__(self, d, h):
            self.ln1 = nn.LayerNorm(d)
            self.attn = LinearAttn(d, h)     # <- remplace SoftmaxAttn
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.SiLU(), nn.Linear(4*d, d))
        def forward(self, x):
            x = x + self.attn(self.ln1(x))   # causal par construction
            return x + self.mlp(self.ln2(x))

Regles mesurees :
  - T >= ~1000 : la lineaire gagne, l'ecart croit avec T (O(N^2) vs O(N))
  - T court (<500) : gardez softmax, ou alternez les couches (hybride)
""")
