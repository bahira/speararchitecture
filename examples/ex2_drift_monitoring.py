"""Exemple 2 - monitoring de drift en production avec W1 exact O(n).

Pattern directement copiable : distribution de reference figee, seuil
auto-calibre sur batchs sains, alertes temps reel sur les batchs entrants.

Run: python ex2_drift_monitoring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from spear_activations import exact_wasserstein1


class DriftMonitor:
    """Detecteur de drift par distance de Wasserstein-1 exacte.

    check(batch) -> dict(w1, thr, alert). Copiez cette classe telle quelle.
    """

    def __init__(self, ref_samples, bins=96, calib_batches=30, k=6.0):
        ref = np.asarray(ref_samples, np.float64).ravel()
        lo, hi = ref.mean() - 6 * ref.std(), ref.mean() + 6 * ref.std()
        self.edges = np.linspace(lo, hi, bins + 1)
        h, _ = np.histogram(np.clip(ref, lo, hi), bins=self.edges)
        self.ref_p = h / h.sum()
        self.calib_hist, self.calib_batches, self.k = [], calib_batches, k
        self.thr = None
        self.n_alerts = 0

    def check(self, batch):
        b = np.asarray(batch, np.float64).ravel()
        h, _ = np.histogram(np.clip(b, self.edges[0], self.edges[-1]),
                            bins=self.edges)
        q = h / max(h.sum(), 1)
        w1 = exact_wasserstein1(self.ref_p, q)

        if self.thr is None:
            if len(self.calib_hist) < self.calib_batches:
                self.calib_hist.append(w1)
                return {"w1": w1, "thr": None, "alert": False}
            arr = np.array(self.calib_hist)
            self.thr = max(float(arr.mean() + self.k * arr.std()), 1e-9)
        alert = bool(w1 > self.thr)
        self.n_alerts += alert
        return {"w1": w1, "thr": self.thr, "alert": alert}


# ---------------- demo : un service ML et ses deux pannes -------------------
# Phases (le piege classique = calibrer pendant le drift ; ici la calibration
# consomme 30 batchs SAINS, les pannes n'arrivent qu'apres) :
#   batchs 00-39 : regime nominal (les 30 premiers servent a calibrer le seuil)
#   batchs 40-59 : panne 1 - la moyenne glisse progressivement (+1.5 sigma max)
#   batchs 60-79 : panne corrigee... puis variance qui explose
rng = np.random.default_rng(42)
monitor = DriftMonitor(rng.standard_normal(50_000))
print(f"seuil auto-calibre apres {monitor.calib_batches} batchs sains...")

timeline, first_alert, false_alarms = [], None, 0
DRIFT_AT, FIX_AT = 40, 60
for t in range(80):
    if t < DRIFT_AT:
        x = rng.standard_normal(256)
    elif t < FIX_AT:
        x = rng.standard_normal(256) + min(1.5, (t - DRIFT_AT + 1) * 0.15)
    else:
        x = rng.standard_normal(256) * (1.0 + (t - FIX_AT) * 0.05)

    r = monitor.check(x)
    tag = "ALERT" if r["alert"] else ("calib" if r["thr"] is None else "ok")
    if r["alert"]:
        if first_alert is None:
            first_alert = t
        if t < DRIFT_AT:
            false_alarms += 1
    timeline.append(tag)
    print(f"batch {t:02d}  W1={r['w1']:7.3f}  "
          f"{'thr=%.3f' % r['thr'] if r['thr'] else 'thr=   - '}  {tag}")

print("\n=== resume ===")
print("timeline :", " ".join(timeline[:20]), "|", " ".join(timeline[20:40]),
      "|", " ".join(timeline[40:60]), "|", " ".join(timeline[60:]))
print(f"fausses alarmes en regime sain : {false_alarms}")
lat = None if first_alert is None else first_alert - DRIFT_AT
print(f"panne 1 (moyenne) injectee au batch {DRIFT_AT} -> premiere alerte batch "
      f"{first_alert} (latence {lat} batchs)")

