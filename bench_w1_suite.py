"""Bench suite : 3 cas d'usage de exact_wasserstein1 (W1 exacte O(n))."""
import time

import numpy as np
from scipy.stats import lognorm, norm, wasserstein_distance

from spear_activations import exact_wasserstein1


def hist_of(x, edges):
    h, _ = np.histogram(x, bins=edges)
    return h / h.sum()


def scenario1():
    print("=" * 62)
    print("SCENARIO 1 - La moyenne ment, W1 ne ment pas")
    print("=" * 62)
    rng = np.random.default_rng(42)
    a = lognorm(s=0.35, scale=100.0).rvs(size=200_000, random_state=rng)
    b = lognorm(s=0.55, scale=102.0).rvs(size=200_000, random_state=rng)

    mean_a, mean_b = a.mean(), b.mean()
    p99_a, p99_b = np.percentile(a, 99), np.percentile(b, 99)
    hi = max(a.max(), b.max()) * 1.2
    edges = np.linspace(0.0, hi, 65)
    w1_ms = exact_wasserstein1(hist_of(a, edges), hist_of(b, edges)) * (hi / 64)

    print(f"{'metrique':22s} {'variante A':>12s} {'variante B':>12s} {'ecart':>8s}")
    print(f"{'-' * 62}")
    print(f"{'moyenne (ms)':22s} {mean_a:12.2f} {mean_b:12.2f} {(mean_b - mean_a) / mean_a * 100:+7.2f}%")
    print(f"{'p99 (ms)':22s} {p99_a:12.2f} {p99_b:12.2f} {(p99_b - p99_a) / p99_a * 100:+7.2f}%")
    print(f"{'W1 histogrammes':22s} {w1_ms:12.2f} {'':>12s}")
    print()
    print(f">> Moyenne : +{(mean_b - mean_a) / mean_a * 100:.2f}% seulement -> invisible en prod.")
    print(f">> Mais p99 : +{(p99_b - p99_a) / p99_a * 100:.1f}% et W1 = {w1_ms:.1f} ms de masse dequeue.")
    print(">> La moyenne ment ; W1 localise la regression de queue que la moyenne cache.")


def scenario2():
    print("=" * 62)
    print("SCENARIO 2 - Ou est le drift ? (localisation par feature)")
    print("=" * 62)
    rng = np.random.default_rng(7)
    F = 12
    means = rng.normal(0.0, 3.0, size=F)
    sigmas = rng.uniform(0.5, 1.5, size=F)
    drift_idx = [2, 7, 10]
    modes = {2: "shift +0.8s", 7: "sigma x1.5", 10: "shift +0.8s"}

    ref = rng.normal(means, sigmas, size=(20_000, F))
    prod = rng.normal(means, sigmas, size=(5_000, F))
    prod[:, 2] += 0.8 * sigmas[2]
    prod[:, 10] += 0.8 * sigmas[10]
    prod[:, 7] = means[7] + (prod[:, 7] - means[7]) * 1.5

    rows = []
    for j in range(F):
        lo, hi = means[j] - 6 * sigmas[j], means[j] + 6 * sigmas[j]
        edges = np.linspace(lo, hi, 65)
        w1_bins = exact_wasserstein1(hist_of(ref[:, j], edges), hist_of(prod[:, j], edges))
        rows.append((j, w1_bins, j in drift_idx))

    rows.sort(key=lambda r: -r[1])
    top3 = sorted(r[0] for r in rows[:3])
    ok = top3 == sorted(drift_idx)

    print(f"{'rang':4s} {'feature':7s} {'W1 (bins)':>10s} {'derivee ?':>10s} {'mode':>14s}")
    print("-" * 62)
    for rank, (j, w1, drifted) in enumerate(rows, 1):
        print(f"{rank:<4d} {j:<7d} {w1:>10.3f} {'OUI' if drifted else 'non':>10s} {modes.get(j, ''):>14s}")
    print()
    print(f">> top-3 du ranking = {top3}, features injectees = {sorted(drift_idx)}")
    print(f">> top-3 retrouve : {'OUI' if ok else 'NON'}")


def scenario3():
    print("=" * 62)
    print("SCENARIO 3 - Vitesse : notre W1 vs scipy")
    print("=" * 62)
    rng = np.random.default_rng(2024)
    D = 2048
    p = rng.random(D); p /= p.sum()
    q = rng.random(D); q /= q.sum()
    pos = np.arange(float(D))
    exact_wasserstein1(p, q)
    wasserstein_distance(pos, pos, p, q)

    def bench(fn, n=50):
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(t))

    ours = bench(lambda: exact_wasserstein1(p, q))
    sp = bench(lambda: wasserstein_distance(pos, pos, p, q))
    v_ours = exact_wasserstein1(p, q)
    v_sp = wasserstein_distance(pos, pos, p, q)

    print(f"D = {D} bins, 50 repetitions chacun")
    print(f"{'implementation':28s} {'mediane (ms)':>13s}")
    print("-" * 62)
    print(f"{'exact_wasserstein1 (SPEAR)':28s} {ours:13.3f}")
    print(f"{'scipy.wasserstein_distance':28s} {sp:13.3f}")
    print(f"{'ratio scipy/nous':28s} {sp / ours:13.1f}x")
    print(f"valeur identique : nous={v_ours:.6f}  scipy={v_sp:.6f}")


if __name__ == "__main__":
    scenario1()
    print()
    scenario2()
    print()
    scenario3()
