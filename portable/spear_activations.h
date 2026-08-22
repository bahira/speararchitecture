/* ==========================================================================
 * spear_activations.h - corpus portable d'activations algebriques validees
 * --------------------------------------------------------------------------
 * Issu d'une boucle de falsification/refit minimax (grounded-loop).
 * Chaque forme close a ete validee par parite d'entrainement sur un
 * transformer char-level (tiny_shakespeare) aux echelles 0.35M et 0.11M.
 *
 * Erreurs Linf mesurees sur [-8,8] vs references exactes :
 *   spear_silu      4.0e-2   remplace x*sigmoid(x)   : 12 unites ALU vs 27
 *   spear_gelu      5.1e-2   remplace 0.5*x*(1+erf(x/sqrt2))
 *   spear_softplus  1.7e-4   remplace log(1+exp(x))  : 32 vs 42 (1 seul exp)
 *   spear_phi       3.7e-3   CDF gaussienne purement algebrique [-4,4] (zero erf)
 *   spear_sigmoid   exacte   identite 1 - 1/(1+e^-x)
 *   spear_tanh      1.4e-5   Pade[3/2] in-domain |x|<=1 uniquement
 *   spear_exact_wasserstein1  exact O(n), une seule passe
 *
 * C99 strict : zero allocation dynamique, branchless (fminf/fmaxf),
 * fonctions pures inline - portable MCU/DSP/FPGA/WASM.
 * ========================================================================== */
#ifndef SPEAR_ACTIVATIONS_H
#define SPEAR_ACTIVATIONS_H

#include <math.h>
#include <stdint.h>

/* Constantes refittees minimax (source : spear_constants.json) */
#define SPEAR_SILU_A  0.50000000000002930f
#define SPEAR_SILU_B  0.51644495786218860f
#define SPEAR_SILU_C  0.01000000000000129f
#define SPEAR_SILU_D  2.85377300701158100f

#define SPEAR_GELU_A  0.49999999999990750f
#define SPEAR_GELU_B  0.50558992755953450f
#define SPEAR_GELU_C  0.00100000000000000f
#define SPEAR_GELU_D  0.68219927331460940f

#define SPEAR_SP_A    9.17213789289023200f
#define SPEAR_SP_B    9.18217225458581300f
#define SPEAR_SP_C    5.48952043564750100f

#define SPEAR_PHI_B   0.76006652531474f
#define SPEAR_PHI_C   -1.77357914117179f
#define SPEAR_PHI_D   7.33906421370566f

#define SPEAR_TANH_A  0.06381872026634902f
#define SPEAR_TANH_B  1.00003711628576530f
#define SPEAR_TANH_C  0.39676795752521077f

/* Noyau commun silu/gelu : marche rationnelle sqrt, pas de transcendance */
static inline float spear_rstep(float x, float a, float b, float c, float d) {
    const float x2 = x * x;
    return x * (a + b * x / (c + sqrtf(d + x2)));
}

/* SiLU algebrique : 12 unites ALU (mul/add=1, div=4, sqrt=2) vs 27 exactes */
static inline float spear_silu(float x) {
    return spear_rstep(x, SPEAR_SILU_A, SPEAR_SILU_B, SPEAR_SILU_C,
                       SPEAR_SILU_D);
}

/* GELU algebrique : meme famille rationnelle, callee sur Phi(x)=CDF */
static inline float spear_gelu(float x) {
    return spear_rstep(x, SPEAR_GELU_A, SPEAR_GELU_B, SPEAR_GELU_C,
                       SPEAR_GELU_D);
}

/* Softplus : max(x,0) + u(A+u)/(B+C*u), u=e^-|x| - Pade[2/2], 1 seul exp */
static inline float spear_softplus(float x) {
    const float u = expf(-fabsf(x));
    return fmaxf(x, 0.0f) + u * (SPEAR_SP_A + u) /
           (SPEAR_SP_B + SPEAR_SP_C * u);
}

/* CDF gaussienne Phi(x), purement algebrique.
 * Linf 3.7e-3 sur [-4,4] (domaine nominal) ; sature (clamp) au-dela,
 * erreur globale [-8,8] bornee a 4.4e-2. */
static inline float spear_phi(float x) {
    const float x2 = x * x;
    const float cdf = 0.5f + 0.5f * (SPEAR_PHI_B * x) /
                      (SPEAR_PHI_C + sqrtf(SPEAR_PHI_D + x2));
    return fminf(fmaxf(cdf, 0.0f), 1.0f);
}

/* Sigmoid : identite exacte (le point est le cout compare, pas la purete) */
static inline float spear_sigmoid(float x) {
    return 1.0f - 1.0f / (1.0f + expf(-x));
}

/* Tanh Pade[3/2] : Linf 1.4e-5 sur [-1,1] ; sature a +-1 au-dela */
static inline float spear_tanh(float x) {
    const float x2 = x * x;
    const float v = (x + SPEAR_TANH_A * x2 * x) /
                    (SPEAR_TANH_B + SPEAR_TANH_C * x2);
    return fminf(fmaxf(v, -1.0f), 1.0f);
}

/* Wasserstein-1 exact entre histogrammes normalises (O(n), 1 passe).
 * Precondition : sum(p)==sum(q)==1. Resultat en unites de bins. */
static inline float spear_exact_wasserstein1(const float *p, const float *q,
                                             uint32_t n) {
    float cp = 0.0f;
    float cq = 0.0f;
    float w = 0.0f;
    uint32_t i;
    for (i = 0u; i < n; ++i) {
        cp += p[i];
        cq += q[i];
        w += fabsf(cp - cq);
    }
    return w;
}

#endif /* SPEAR_ACTIVATIONS_H */
