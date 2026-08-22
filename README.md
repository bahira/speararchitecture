# SPEAR Architecture

> **Opérateurs algébriques validés pour le ML + toolkit LLM hyper-léger.** Chaque formule est
> falsifiée, refittée minimax, puis validée par **parité d'entraînement réelle** — pas de claims
> non mesurés.

![license](https://img.shields.io/badge/license-MIT-green) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![lang](https://img.shields.io/badge/backends-PyTorch%20%7C%20C99%20%7C%20JS-orange) ![status](https://img.shields.io/badge/status-research%20validated-brightgreen)

Ce dépôt est né de la vérification d'un article (« GROUNDED-SPEAR V1200 ») prétendant que des
formes closes algébriques remplacent avantageusement les kernels standards (SiLU/GELU/softmax/
Sinkhorn…). Verdict en deux lignes : **les structures sont souvent bonnes, les constantes publiées
étaient fausses** — ce repo livre les vraies constantes, les preuves d'entraînement et les
benchmarks honnêtes (y compris les gains qui n'ont PAS été retrouvés).

---

## Table des matières
1. [Gains validés](#1-gains-validés)
2. [Falsifications — le paper avait tort](#2-falsifications)
3. [Quickstart](#3-quickstart)
4. [Intégration dans votre projet](#4-intégration)
5. [Benchmarks complets](#5-benchmarks)
6. [Méthodologie](#6-méthodologie)
7. [Limites honnêtes](#7-limites)
8. [Layout du repo](#8-layout)
9. [Reproduction intégrale](#9-reproduction)

---

## 1. Gains validés

Classés par ampleur mesurée sur machine réelle (CPU 4 cœurs, PyTorch 2.9 CPU, gcc 15.2) :

| # | Gain | Mesure | Où ça compte |
|---|---|---|---|
| 1 | **Wasserstein-1 exact O(n)** | **×28 vs scipy** (accord 2.4e-14) · ×642 000 vs Sinkhorn log-domain (qui reste faux à 76 %) | monitoring drift prod, vision, audio |
| 2 | **Attention linéaire O(N)** | crossover T≈1000 → **×1.9 @2048**, **×5.1 @8192** (hd=16 ; ×8.5 avec B adapté) | contextes longs |
| 3 | **Activations sans exp/erf** | 12 unités ALU vs 27–34 (**×2.25–2.8**) · facture d'activations MLP **−56 %** · parité d'entraînement prouvée à 0.35M et 0.11M params | MCU/DSP/WASM/kernels custom |
| 4 | **Φ(x) CDF purement algébrique** | L∞ **3.7e-3** sur [-4,4], zéro `erf` | scoring statistique embarqué |
| 5 | **Moniteur de drift W1** | 0 fausse alarme · détection en 1 batch après injection · seuil auto-calibré | MLOps temps réel |
| 6 | int8 dynamique | mémoire **÷2.6–3.2**, qualité intacte (+0.0006 val loss) | stockage/edge (latence CPU : voir §5.4) |
| 7 | **SDPA fusé** | entraînement **×4.22** (441 vs 1863 ms/step), identique à 6e-7 | tout training softmax-attention |
| 8 | **Ternaire natif STE** | val +0.125 seulement vs fp32, mémoire ÷5.7 — le post-training échouait, l'entraînement dedans passe | edge/stockage extrême |

## 2. Falsifications

Tout ce qui a été testé et **rejeté**, avec la preuve :

| Claim du paper d'origine | Réalité mesurée ici |
|---|---|
| SiLU « record 8.2e-4 » (`x·(0.501+0.589x/(0.83+√(1+x²)))`) | erreur réelle **2.5e-1** (~300× le claim). Plafond structurel de cette famille : ~4e-2 après refit minimax |
| GELU linéaire-cap « 5.3e-4 » | erreur réelle **8.8e-2**, et **dégrade l'entraînement** (val 3.234 vs 3.013) → rejeté au profit d'une famille rationnelle-√ découverte dans la boucle |
| β(t) diffusion exponentiel-cosinus | **0 % des valeurs dans (0,1)** — schedule invalide |
| β(t) raffiné `−0.496cos(3.162t)+0.501` | bornes OK mais ≠ cosine schedule (écart max **0.96**) |
| Tanh Padé[3/2] constantes publiées | 0.23 d'erreur même sur [-1,1] ; après refit : **1.4e-5** in-domain uniquement |
| Blur `exp(cos x)` | n'est qu'une gaussienne déguisée in-domain (PSNR 71 dB) ; la forme raffinée EST la gaussienne (diff kernel 1.6e-3) |
| Speedups torch eager ×2.43 / ×6.57 | non reproductibles en eager — ATen fusionné gagne (×0.35–0.81). Le gain est op-level/kernel custom. En **JS/V8 en revanche : ×2.96 wall-clock réel** |

La boucle a aussi attrapé son propre bug : une première version de Φ fittée contre la sigmoïde
logistique au lieu de la vraie CDF — détecté par l'audit croisé C↔Python, corrigé.

## 3. Quickstart

```bash
pip install numpy scipy torch        # seules dépendances

# self-checks (aucun téléchargement requis)
python spear_activations.py          # parité Python : 6/6 PASS
cd portable && gcc -std=c99 -Wall -Wextra -pedantic -O2 test_spear.c -o test_spear -lm && ./test_spear   # AUDIT PASS
node ../portable/spear_kernels.js    # corpus JS (module)

# LLM char-level hyper-léger
python spear_llm.py fetch                                        # tiny_shakespeare (~1 MB)
python spear_llm.py train --act spear_silu --d 64 --nl 2         # 110k params, ~2 min CPU
python spear_llm.py sample --ckpt out/spear_silu_d64n2.pt
```

## 4. Intégration

Trois chemins, tous un-fichier-copié (voir [`examples/`](examples/) pour les démos exécutables) :

### 4.1 Activations dans un modèle PyTorch
```python
import spear_activations as sa          # copiez le fichier, constantes embarquées
model = nn.Sequential(..., sa.SpearSiLU(), ...)   # remplace nn.SiLU()/nn.GELU()
```

### 4.2 Drift monitoring en production
```python
from examples.ex2_drift_monitoring import DriftMonitor
mon = DriftMonitor(ref_samples)                 # référence figée à l'entraînement
r = mon.check(new_batch)                        # {"w1": ..., "alert": True/False}
```

### 4.3 Attention linéaire long-contexte
```python
from spear_llm import LinearAttn                # ou SoftmaxAttn
x = x + LinearAttn(d, h)(self.ln1(x))           # causal par construction
```
Règle mesurée : T ≥ ~1000 → gagne ; T court → softmax ou hybride.

### 4.4 C99 / JS embarqué
```c
#include "spear_activations.h"                  // header unique, malloc=0, branchless
float y = spear_gelu(x);
float w = spear_exact_wasserstein1(p, q, n);    // O(n), une passe
```
```js
const SPEAR = require("./portable/spear_kernels.js");   // navigateur : window.SPEAR
SPEAR.silu(x); SPEAR.wasserstein1(p, q);
```

## 5. Benchmarks complets

Tous les chiffres ci-dessous ont été produits sur la machine de développement
(4 cœurs, Windows, torch 2.9.1 CPU, gcc 15.2 MinGW). Scripts inclus, reproductibles §9.

### 5.1 Parité des opérations (audit croisé C99 ↔ Python)

| Op | Référence | L∞ mesuré | Domaine nominal | ALU/elem (ref → spear) |
|---|---|---|---|---|
| `spear_silu` | x·sigmoid(x) | **4.02e-2** | [-8,8] | 27 → 12 |
| `spear_gelu` | x·Φ(x) (erf) | **5.10e-2** | [-8,8] | 27–34 → 12 |
| `spear_softplus` | log(1+eˣ) | **1.70e-4** | [-6,6] | 42 → 32 |
| `spear_phi` | Φ(x)=½(1+erf(x/√2)) | **3.65e-3** | [-4,4] (clamp global ≤4.4e-2) | erf(~27) → 11 |
| `spear_tanh` | tanh(x) | **1.44e-5** | [-1,1] strict | 21 → 9 |
| `spear_exact_wasserstein1` | scipy | **2.4e-14 rel.** | partout | itératif → O(n) |

Unités ALU du modèle de coût : mul/add=1, div=4, sqrt=2, exp/log≈20.
Facture par token (d=96, 3 blocs) : matmuls 663k unités (dominant), activations 31.1k → 13.8k (**−56 %**
d'activations, −2.6 % end-to-end), attention softmax 38.4k vs linéaire ~31.7k mais O(N²)→O(N).

### 5.2 Parité d'entraînement (tiny_shakespeare, char-level, 500 steps, seed fixe)

| Config | Activation | val loss | ppl |
|---|---|---|---|
| 0.35M (d96·L3) | F.silu | 2.9873 | 19.83 |
| 0.35M | **SiLU algébrique** | **2.9584** | **19.27** |
| 0.35M | F.gelu | 3.0128 | 20.35 |
| 0.35M | ~~GELU linéaire-cap~~ | 3.2342 ❌ | 25.38 |
| 0.35M | **GELU rationnel √** | **2.9753** | **19.59** |
| 0.11M (d64·L2) | F.silu | 3.0309 | 20.72 |
| 0.11M | **SiLU algébrique** | **3.0122** | **20.33** |
| 0.35M | **Ternaire STE natif** (`--ternary`) | **3.1370** | 23.03 |
| 0.35M | attention hybride softmax/linéaire | 3.1208 | 22.67 |
| 0.35M | attention linéaire + décroissance RetNet | 3.3471 ❌ | 28.42 |

→ Les formes sans transcendante s'entraînent aussi bien que les exactes, aux deux échelles.
→ Le ternaire entraîné nativement (STE absmean) tient à +0.125 du fp32 — mémoire packée ÷5.7.

### Vitesse d'entraînement

`F.scaled_dot_product_attention` (kernel fusé causal) vs attention manuelle, même modèle :

| Chemin | ms/step | gain |
|---|---|---|
| manuel fp32 | 1863 | ×1 |
| **SDPA fp32** | **441** | **×4.22** |
| bf16 autocast | abandonné | pas de bf16 natif sur ce CPU |

Écart logits manuel↔SDPA à poids identiques : 6e-7. SDPA est **activé par défaut** dans
`spear_llm.py train` (`--no-sdpa` pour l'ancien chemin).

### 5.3 Attention softmax vs linéaire (fwd+bwd)

| T | hd=24 speedup | hd=16 speedup |
|---|---|---|
| 128–512 | ×0.48–0.75 | ×0.75 |
| 1024–2048 | ×1.01–1.08 | ×1.91 |
| 4096 | ×1.91–2.34 | **×3.73** |
| 8192 | ×4.06 | **×5.06–8.46** |

### 5.4 Quantification (modèle hyper-léger)

| Variante | disque | Δval | latence tok/s ratio |
|---|---|---|---|
| fp32 0.11M | 449 KB | — | 1.00 |
| int8 dyn 0.11M | **171 KB (÷2.62)** | +0.020 | 0.84 (plus lent) |
| fp32 0.35M | 1397 KB | — | 1.00 |
| int8 dyn 0.35M | **440 KB (÷3.17)** | **+0.0006** | 0.61 (plus lent) |
| ternaire naïf | 64–146 KB packés (÷7–9.6) | +0.50/+0.80 ❌ | — (rejeté sans QAT) |

Honnêteté : sur ce CPU x86 l'int8 gagne la mémoire, perd la latence (déquant) ; sur ARM/int8-hw le
verdict s'inverse généralement.

### 5.5 Wasserstein-1 exact (D=256, histogrammes aléatoires)

| Méthode | valeur | temps | erreur |
|---|---|---|---|
| **exacte O(n) (ce repo)** | 3.778550 | **0.09 ms** | 0 (par définition) |
| scipy `wasserstein_distance` | 3.778550 | 2.49 ms | 2.4e-14 → **×28 plus lente** |
| Sinkhorn log-domain (3000 it) | 0.919080 | 57.7 s | **76 %** → ×642 000 plus lente ET fausse |

### 5.6 Moniteur de drift (`drift_monitor.py`, flux simulé 120 batchs)

Seuil auto-calibré 2.37 bins sur 25 batchs propres · **0 fausse alarme** · drift injecté t=40,
détecté t=44 (1024 échantillons) · 76/80 ticks post-drift en alerte · W1 maison ×6.9 vs scipy cumulé.

### 5.7 KV-cache (éviction 50 %, masse d'attention future retenue)

random ≈0.50 · fenêtre ≈0.40 · S seul 0.54–0.72 · **H2O (attention seule) 0.64–0.75** ·
additif `4S+A+1.5R` 0.60–0.69 · multiplicatif `(A+R)(1+3S)` 0.58–0.67.
Les règles tri-dimensionnelles atteignent le claim ~67 % du paper mais ne battent pas H2O pur.

## 6. Méthodologie

Chaque affirmation passe une boucle type *grounded-loop* : formulation falsifiable → mesure
exécutable (L≥3) → refit si échec → adversarial (audit croisé C↔Python↔JS, seeds fixés) → verdict
typé consigné. Le ledger complet des 12 itérations, gaps typés (G-FACT/G-PERF/G-SPEC…) et verdicts
vit dans [`loop_state.json`](loop_state.json).

## 7. Limites

- Modèles char-level de démonstration : ppl 19–21 = semi-charabia. Ce sont des **bancs de preuve**, pas des générateurs.
- Pas de gain latence des activations en torch eager (win réel : JS ×2.96 mesuré, kernels custom, MCU).
- Attention linéaire pure : qualité courte-T non résolue — décroissance RetNet-style sans effet (3.347) ;
  l'hybride alterné récupère ~60 % de l'écart (3.121). Piste restante : delta-rule / chunk-wise.
- Ternaire : OK entraîné nativement (STE), rejeté en post-training seul.
- bf16 autocast : sans intérêt sur CPU sans support natif (émulation ×20 plus lente).
- Mono-seed, CPU-only pour l'entraînement.

## 8. Layout

```
spear_activations.py     wrapper portable torch/numpy + self-check      <- point d'entrée principal
spear_constants.json     constantes refittées (source de vérité unique)
portable/                spear_activations.h (C99) + audit + spear_kernels.js
examples/                5 démos exécutables (torch swap, drift prod, attention, C, node)
spear_llm.py             transformer char-level complet (train/sample/kv, 4 types d'attention)
drift_monitor.py         moniteur de drift standalone
bench_attn/light/ot.py   benchmarks (attention, quantification+coût, OT)
spear_ops.py refine_*.py pipeline de falsification/refit
loop_state.json          ledger grounded-loop
out/results.json         résultats d'entraînement (checkpoints .pt non versionnés)
```

## 9. Reproduction

```bash
python spear_activations.py                                   # §5.1 python
cd portable && gcc -std=c99 -Wall -Wextra -pedantic -O2 test_spear.c -o test_spear -lm && ./test_spear
node ../portable/spear_kernels.js                             # §5.1 js
cd .. && python bench_ot.py                                   # §5.5
python drift_monitor.py                                       # §5.6
python spear_llm.py kv                                        # §5.7 (après 1 train)
python spear_llm.py train --act silu --steps 500              # §5.2 bras référence (~4 min)
python spear_llm.py train --act spear_silu --steps 500        # §5.2 bras algébrique
python bench_attn.py                                          # §5.3
python bench_light.py                                         # §5.4
python examples/ex1_activations.py                            # …et examples/README.md
```

## Licence

MIT — voir [LICENSE](LICENSE).
