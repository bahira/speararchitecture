/**
 * Exemple 5 - corpus JS zero-dependance (node ou navigateur).
 *
 * Run node :  node ex5_js_kernels.js
 * Navigateur: <script src="../portable/spear_kernels.js"></script> puis SPEAR.*
 */
const SPEAR = require("../portable/spear_kernels.js");

// --- activations -------------------------------------------------------
const sig = (x) => 1 / (1 + Math.exp(-x));
console.log(`spear.silu(1.0)      = ${SPEAR.silu(1).toFixed(6)}   exact ${(1 * sig(1)).toFixed(6)}`);
console.log(`spear.softplus(2.0)  = ${SPEAR.softplus(2).toFixed(6)}   exact ${Math.log(1 + Math.exp(2)).toFixed(6)}`);
console.log(`spear.phi(1.96)      = ${SPEAR.phi(1.96).toFixed(6)}   (CDF sans erf)`);

// --- distance entre histogrammes (drift, vision, audio) ----------------
const ref = [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05];
const batch = [0.02, 0.06, 0.12, 0.18, 0.24, 0.19, 0.13, 0.06];
console.log(`W1(reference,batch) = ${SPEAR.wasserstein1(ref, batch).toFixed(4)} bins`);

// --- micro-benchmark 1M evals -------------------------------------------
{
  const N = 1_000_000;
  let t0 = performance.now();
  let s = 0;
  for (let i = 0; i < N; i++) s += SPEAR.silu((i % 1000) / 100 - 5);
  const tSpear = performance.now() - t0;
  t0 = performance.now();
  s = 0;
  for (let i = 0; i < N; i++) s += ((i % 1000) / 100 - 5) * sig((i % 1000) / 100 - 5);
  console.log(`silu  : spear ${tSpear.toFixed(1)} ms vs exact ${ (performance.now()-t0).toFixed(1)} ms pour ${N} evals`);
}
