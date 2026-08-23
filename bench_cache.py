"""Bench autonome : génération KV-cache vs naïve (regen totale O(n²)).
Charge out/spear_silu.pt, génère n_new tokens avec les deux chemins (même seed),
imprime ms/token, speedup et la parité full-forward vs prefill.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from spear_llm import OUT, kv_parity, load_ckpt, load_data


def naive_gen(m, Tmax, prompt, n):
    idx = prompt.clone()
    with torch.no_grad():
        for _ in range(n):
            logits, _ = m(idx[:, -Tmax:])
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits[:, -1, :], -1), 1)], 1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "spear_silu.pt"))
    ap.add_argument("--n-new", type=int, default=300)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    _, va, _, _ = load_data()
    m, cfg = load_ckpt(a.ckpt)
    prompt = va[:a.prompt_len].unsqueeze(0)
    n = a.n_new

    # warmup des deux chemins (JIT/thread pool) hors chronomètre
    m.generate(prompt, 8)
    naive_gen(m, cfg["T"], prompt, 8)

    def timed(fn):
        """Best-of-3 (les timings Windows sont bruités : outliers >30x vus),
        arrêt anticipé si le budget global explose."""
        times = []
        while len(times) < 3:
            torch.manual_seed(a.seed)
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
            if sum(times) > 75:
                break
        return times

    tn = timed(lambda: naive_gen(m, cfg["T"], prompt, n))
    tc = timed(lambda: m.generate(prompt, n))
    t_naive, t_cache = min(tn), min(tc)

    diff = kv_parity(m, prompt[:, -cfg["T"]:])
    print(f"ckpt={os.path.basename(a.ckpt)} d={cfg['d']} nl={cfg['nl']} h={cfg['h']} "
          f"T={cfg['T']} prompt={a.prompt_len} n_new={n} threads=4")
    print(f"{'method':8s} {'best_s':>8s} {'ms/token':>9s}   all_runs_s")
    print(f"{'naive':8s} {t_naive:8.2f} {t_naive/n*1000:9.2f}   "
          f"{['%.2f' % t for t in tn]}")
    print(f"{'cache':8s} {t_cache:8.2f} {t_cache/n*1000:9.2f}   "
          f"{['%.2f' % t for t in tc]}")
    print(f"SPEEDUP (best): {t_naive/t_cache:.2f}x")
    print(f"parity max|diff| (full fwd vs prefill incrémental, L={prompt.size(1)}) = {diff:.3e}"
          f"  ({'OK' if diff < 1e-4 else 'FAIL > 1e-4'})")
    if a.prompt_len + n > cfg["T"]:
        print("note: fenêtre saturée — le cache tronque le front des K/V (approximation),"
              " les timings restent comparables")


if __name__ == "__main__":
    main()
