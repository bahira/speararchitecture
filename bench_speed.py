"""Lab vitesse d'entraînement CPU : trouver la config la plus rapide honnête.
Compare : attention manuelle vs SDPA fusé, fp32 vs autocast bfloat16.
Run: python bench_speed.py
"""
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import spear_llm as sl

torch.set_num_threads(4)


def make(attn="softmax"):
    act = sl.make_acts(sl.consts())["silu"]
    return sl.GPT(65, d=96, nl=3, h=4, T=128, act=act, attn=attn)


def patch_sdpa(model):
    """Remplace le coeur softmax par F.scaled_dot_product_attention (fusé)."""
    for blk in model.blocks:
        blk.attn.use_sdpa = True


def timed_steps(m, opt, B=16, T=128, reps=20):
    x = torch.randint(0, 65, (B, T))
    y = torch.randint(0, 65, (B, T))
    m.train()
    for _ in range(3):
        loss = m(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = m(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / reps
    return dt


def main():
    import sys
    B = 16
    print(f"B={B} T=128 d=96 L=3\n", flush=True)

    # 1) baseline actuel : attention manuelle fp32
    m1 = make()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    d1 = timed_steps(m1, o1)
    print(f"fp32 manuel   : {d1*1e3:6.0f} ms/step  ({1/d1:5.2f} it/s)", flush=True)

    # 2) SDPA fp32
    m2 = make()
    patch_sdpa(m2)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    d2 = timed_steps(m2, o2, reps=15)
    print(f"fp32 SDPA     : {d2*1e3:6.0f} ms/step  ({1/d2:5.2f} it/s)  x{d1/d2:.2f}", flush=True)

    # 3) SDPA + autocast bf16 — garde-fou si le CPU n'a pas de bf16 natif
    m3 = make()
    patch_sdpa(m3)
    o3 = torch.optim.AdamW(m3.parameters(), lr=1e-3)
    x = torch.randint(0, 65, (B, 128))
    y = torch.randint(0, 65, (B, 128))
    m3.train()
    t0 = time.perf_counter()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = m3(x, y)[1]
    loss.backward()
    o3.step()
    probe = time.perf_counter() - t0
    if probe > 8.0:
        print(f"bf16 SDPA     : ABANDON ({probe:.1f}s pour 1 step — pas de bf16 natif)", flush=True)
        d3 = None
    else:
        for _ in range(2):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = m3(x, y)[1]
            o3.zero_grad()
            loss.backward()
            o3.step()
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = m3(x, y)[1]
            o3.zero_grad()
            loss.backward()
            o3.step()
        d3 = (time.perf_counter() - t0) / 5
        print(f"bf16 SDPA     : {d3*1e3:6.0f} ms/step  ({1/d3:5.2f} it/s)  x{d1/d3:.2f}", flush=True)

    # sanity : SDPA == manuel à poids identiques ?
    m2.load_state_dict(m1.state_dict())
    m1.eval(); m2.eval()
    with torch.no_grad():
        xs = torch.randint(0, 65, (2, 128))
        diff = (m1(xs)[0] - m2(xs)[0]).abs().max()
    print(f"\nmax|logits manuel - SDPA| (poids identiques) : {diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
