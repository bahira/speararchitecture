# Exemples — démos réelles des artefacts SPEAR

Chaque exemple est autonome et exécutable. Ils montrent **comment intégrer**
les gains mesurés du projet dans votre propre code.

| # | Fichier | Ce que ça montre | Commande |
|---|---|---|---|
| 1 | `ex1_activations.py` | Swap `nn.SiLU()` → `SpearSiLU()` dans un réseau PyTorch : parité d'entraînement + coût élémentaire + CDF sans erf | `python ex1_activations.py` |
| 2 | `ex2_drift_monitoring.py` | **Classe `DriftMonitor` copiable en prod** : référence figée, seuil auto-calibré, alertes temps réel sur deux pannes simulées | `python ex2_drift_monitoring.py` |
| 3 | `ex3_linear_attention.py` | Attention linéaire O(N) dans un bloc standard + benchmark du crossover softmax/linéaire sur votre machine | `python ex3_linear_attention.py` |
| 4 | `ex4_c_integration.c` | Intégration C99 du header portable : activations + W1 dans une boucle embarquée zéro-allocation | `gcc -std=c99 -O2 ex4_c_integration.c -o ex4_c_integration -lm && ./ex4_c_integration.exe` |
| 5 | `ex5_js_kernels.js` | Corpus JS zéro-dépendance sous node : activations, W1, micro-benchmark | `node ex5_js_kernels.js` |

Les exemples Python importent depuis le dossier parent (`spear_activations`,
`spear_llm`) via `sys.path` — lancez-les d'où vous voulez.

## Sorties attendues (machine de référence)

- **ex1** : MSE silu ≈ MSE SpearSiLU (écart < quelques %), phi(±3) exact à ~1e-2.
- **ex2** : 0 fausse alarme pendant la calibration et le régime sain ; alerte
  ~2-4 batchs après l'injection du drift de moyenne ; re-alerte sur l'explosion
  de variance.
- **ex3** : speedup linéaire/softmax ≈ ×1 vers T=512-1024, puis >×4 à T=8192.
- **ex4** : toutes valeurs à ~1e-2 près des références double ; W1 identique.
- **ex5** : mêmes constantes que le header C — parité inter-langages.
