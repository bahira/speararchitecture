"""BT13 benchmark: softmax O(N^2) vs Cauchy-Schwarz linear O(N) attention.
Measures real forward+backward wall-time vs sequence length.
Run: python bench_attn.py
"""
import time

import torch

from spear_llm import SoftmaxAttn, LinearAttn

D, H = 96, 4


def timed(model, T, B, reps=10):
    x = torch.randn(B, T, D)
    model.train()
    for _ in range(2):
        y = model(x)
        y.sum().backward()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = model(x)
        y.sum().backward()
    return (time.perf_counter() - t0) / reps


def main():
    torch.set_num_threads(4)
    print(f"{'T':>6s} {'softmax ms':>11s} {'linear ms':>10s} {'speedup':>8s}  note")
    rows = []
    for T in [128, 256, 512, 1024, 2048, 4096]:
        B = 2
        sm = timed(SoftmaxAttn(D, H), T, B)
        try:
            lin = timed(LinearAttn(D, H, normalize=True), T, B)
            print(f"{T:6d} {sm*1e3:11.1f} {lin*1e3:10.1f} {sm/lin:7.2f}x")
            rows.append(dict(T=T, softmax_ms=sm * 1e3, linear_ms=lin * 1e3, speedup=sm / lin))
        except RuntimeError as e:
            print(f"{T:6d} {sm*1e3:11.1f} {'OOM/slow':>10s}")
            rows.append(dict(T=T, softmax_ms=sm * 1e3, linear_ms=float("nan"), speedup=float("nan")))
    import json
    json.dump(rows, open("bench_attn.json", "w"), indent=2)


if __name__ == "__main__":
    main()