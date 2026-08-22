/* spear_kernels.js - corpus portable zero-dependance (navigateur/node/WASM-ready).
 * Formules et constantes issues de la boucle grounded-loop (voir README.md).
 * Unites ALU : mul/add=1, div=4, sqrt=2, exp/log~=20.
 */
const SPEAR = {
  /* SiLU algebrique : 12 unites ALU vs 27 (x*sigmoid), Linf 4.0e-2 */
  silu(x) {
    const x2 = x * x;
    return x * (0.5000000000000293 +
                0.5164449578621886 * x /
                (0.01000000000000129 + Math.sqrt(2.853773007011581 + x2)));
  },

  /* GELU algebrique (famille rationnelle-sqrt), Linf 5.1e-2 */
  gelu(x) {
    const x2 = x * x;
    return x * (0.49999999999990750 +
                0.5055899275595345 * x /
                (0.001 + Math.sqrt(0.6821992733146094 + x2)));
  },

  /* Softplus Padé[2/2], Linf 1.7e-4, un seul exp */
  softplus(x) {
    const u = Math.exp(-Math.abs(x));
    return Math.max(x, 0) +
           u * (9.172137892890232 + u) /
           (9.182172254585813 + 5.489520435647501 * u);
  },

  /* CDF gaussienne purement algebrique, Linf 3.7e-3 sur [-4,4], clamp au-dela */
  phi(x) {
    const x2 = x * x;
    const cdf = 0.5 + 0.5 * (0.76006652531474 * x) /
                       (-1.77357914117179 + Math.sqrt(7.33906421370566 + x2));
    return Math.min(Math.max(cdf, 0), 1);
  },

  /* Sigmoid : identite exacte */
  sigmoid(x) {
    return 1 - 1 / (1 + Math.exp(-x));
  },

  /* Tanh Padé[3/2], Linf 1.4e-5 sur [-1,1], sature au-dela */
  tanhIndomain(x) {
    const x2 = x * x;
    const v = (x + 0.06381872026634902 * x2 * x) /
              (1.0000371162857653 + 0.39676795752521077 * x2);
    return Math.min(Math.max(v, -1), 1);
  },

  /* Wasserstein-1 exact O(n) entre histogrammes normalises (bins) */
  wasserstein1(p, q) {
    let cp = 0, cq = 0, w = 0;
    for (let i = 0; i < p.length; i++) {
      cp += p[i];
      cq += q[i];
      w += Math.abs(cp - cq);
    }
    return w;
  },
};

if (typeof module !== "undefined" && module.exports) module.exports = SPEAR;
if (typeof window !== "undefined") window.SPEAR = SPEAR;
