"""Hyper-light suite: int8 dynamic quant + ternary weights (BT18) + ALU/SFU cost model.
Cost units per SPEAR codex: mul/add=1, div=4, sqrt=2, exp/log/sin/cos~=20.
Run: python bench_light.py [--ckpt out/spear_silu.pt]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from spear_llm import load_ckpt, load_data, get_batch


@torch.no_grad()
def throughput(m, V, T, reps=30):
    x = torch.randint(0, V, (1, T))
    m.eval()
    for _ in range(3):
        m(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        m(x)
    return reps * T / (time.perf_counter() - t0)


@torch.no_grad()
def val_loss(m, va, B, T, n=10):
    ls = []
    for _ in range(n):
        x, y = get_batch(va, B, T)
        ls.append(m(x, y)[1].item())
    return float(np.mean(ls))


def ternarize_(model):
    """BT18 BitNet-style absmean: W -> clamp(round(W/gamma),-1,1)*gamma on Linear only."""
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.data
                g = w.abs().mean().clamp_min(1e-8)
                mod.weight.data = torch.clamp(torch.round(w / g), -1.0, 1.0) * g


def disk_kb(obj, tmp="_tmp_state.pt"):
    torch.save(obj, tmp)
    kb = os.path.getsize(tmp) / 1024.0
    os.remove(tmp)
    return kb


def cost_table():
    U = {"mul": 1, "add": 1, "div": 4, "sqrt": 2, "exp": 20, "log": 20}
    rows = [
        # (name, units/elem, note)
        ("silu exact  x*sigmoid", U["mul"] * 2 + U["exp"] + U["add"] + U["div"]),
        ("hardswish   x*relu6(x+3)/6", 8),
        ("SPEAR silu  rational-sqrt", 12),
        ("gelu exact  erf-form", 27),
        ("gelu tanh-approx", 34),
        ("SPEAR gelu2 rational-sqrt", 12),
        ("softplus exact log1p(exp)", U["exp"] + U["log"] + 2),
        ("SPEAR softplus (1 exp)", 32),
        ("sigmoid exact", U["exp"] + U["add"] + U["div"] + 1),
    ]
    print(f"{'op':32s} {'units/elem':>10s}")
    for n, u in rows:
        print(f"{n:32s} {u:10d}")

    d, nl, hd, N, h = 96, 3, 24, 128, 4
    matmul_blk = 24 * d * d                      # qkv 6d^2 + proj 2d^2 + fc 16d^2 units
    act_silu_blk = 4 * d * rows[0][1]
    act_spear_blk = 4 * d * 12
    smax_blk = h * N * (U["exp"] + U["add"] + U["div"])
    lin_blk = h * (2 * hd * 7 + 4 * hd * hd)     # norms + outer/cumsum adds (O(N) const)
    tot_mm = nl * matmul_blk
    print(f"\nper-token model bill (d={d},nl={nl},T={N}):")
    print(f"  matmuls           {tot_mm:9,d} units  ({100.0:.0f}% ref)")
    print(f"  acts silu exact   {nl*act_silu_blk:9,d}  | spear {nl*act_spear_blk:9,d}"
          f"  (-{(1-act_spear_blk/act_silu_blk)*100:.0f}% acts, -{nl*(act_silu_blk-act_spear_blk)/tot_mm*100:.1f}% total)")
    print(f"  attn softmax      {nl*smax_blk:9,d}  | linear ~{nl*lin_blk:9,d}  (O(N^2)->O(N))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("out", "spear_silu.pt"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    _, va, _, _ = load_data()
    m, cfg = load_ckpt(a.ckpt)
    B, T, V = 16, cfg["T"], cfg["vocab"]
    out = {"ckpt": a.ckpt}

    vl0 = val_loss(m, va, B, T)
    s0 = throughput(m, V, T)
    b0 = disk_kb(m.state_dict())
    seen, nlin = set(), 0
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            ptr = mod.weight.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                nlin += mod.weight.numel()
    print(f"fp32      : val {vl0:.4f}  tok/s {s0:7.0f}  disk {b0:7.1f} KB")

    mq = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    vlq = val_loss(mq, va, B, T)
    sq = throughput(mq, V, T)
    bq = disk_kb(mq.state_dict())
    print(f"int8-dyn  : val {vlq:.4f}  tok/s {sq:7.0f}  disk {bq:7.1f} KB"
          f"  -> lat int8/fp32 x{sq/s0:.2f}  mem {b0/bq:.2f}x  dval {vlq-vl0:+.4f}")

    mt, _ = load_ckpt(a.ckpt)
    ternarize_(mt)
    vlt = val_loss(mt, va, B, T)
    bt_fp32 = disk_kb(mt.state_dict())
    seen2 = set()
    lin_bytes = sum(mod.weight.numel() * 0.25 for mod in m.modules()
                    if isinstance(mod, nn.Linear)
                    and mod.weight.data_ptr() not in seen2
                    and not seen2.add(mod.weight.data_ptr()))
    other_bytes = sum(t.numel() * t.element_size() for n_, t in m.state_dict().items()
                      if t.data_ptr() not in seen2)
    bt_packed = (lin_bytes + other_bytes) / 1024.0
    print(f"ternary   : val {vlt:.4f}  tok/s {'n/a':>7s}  disk(fp32-stored) {bt_fp32:7.1f} KB"
          f"  packed(theo) {bt_packed:7.1f} KB -> mem {b0/bt_packed:.2f}x  dval {vlt-vl0:+.4f}")

    cost_table()
    out.update(fp32=dict(val=vl0, tok_s=s0, kb=b0), int8=dict(val=vlq, tok_s=sq, kb=bq,
                speedup=s0 / sq), ternary=dict(val=vlt, packed_kb=bt_packed))
    json.dump(out, open("bench_light.json", "w"), indent=2)


if __name__ == "__main__":
    main()
