/**
 * Exemple 4 - integrer le corpus portable dans un projet C99 existant.
 * Zero allocation, zero boucle cache (hors W1), portable MCU/DSP/WASM.
 *
 * Compile + run :
 *   gcc -std=c99 -O2 ex4_c_integration.c -o ex4_c_integration -lm
 *   ./ex4_c_integration.exe
 */
#include <math.h>
#include <stdio.h>

#include "../portable/spear_activations.h"

int main(void) {
    printf("== activations algebriques ==\n");
    {
        const double x = 1.0;
        printf("spear_silu(1.0)     = %.5f | exact %.5f\n",
               (double)spear_silu(1.0f), x / (1.0 + exp(-x)));
    }
    {
        const double x = 0.5;
        const double ref = 0.5 * x * (1.0 + erf(x * 0.70710678118654752));
        printf("spear_gelu(0.5)     = %.5f | exact %.5f\n",
               (double)spear_gelu(0.5f), ref);
    }
    {
        const double x = 2.0;
        printf("spear_softplus(2.0) = %.5f | exact %.5f\n",
               (double)spear_softplus(2.0f), log(1.0 + exp(x)));
    }
    {
        const double x = 1.96; /* seuil z ~ 2 sigma */
        const double ref = 0.5 * (1.0 + erf(x * 0.70710678118654752));
        printf("spear_phi(1.96)     = %.5f | exact %.5f\n",
               (double)spear_phi(1.96f), ref);
    }

    /* ------------------------------------------------------------------
     * Distance entre deux histogrammes de features (drift detection).
     * Precondition : histogrammes normalises sur la MEME grille.
     * Resultat en unites de bins : distance moyenne de transport.
     * ------------------------------------------------------------------ */
    static const float ref[8]   = {0.05f, 0.10f, 0.15f, 0.20f,
                                   0.20f, 0.15f, 0.10f, 0.05f};
    static const float batch[8] = {0.02f, 0.06f, 0.12f, 0.18f,
                                   0.24f, 0.19f, 0.13f, 0.06f};
    const float w1 = spear_exact_wasserstein1(ref, batch, 8u);
    printf("\nW1(reference, batch) = %.4f bins\n", (double)w1);

    /* Boucle chaude type inference : 100k activations sans aucune alloc */
    double acc = 0.0;
    for (int i = 0; i < 100000; ++i) {
        const float x = -6.0f + 12.0f * (float)i / 99999.0f;
        acc += spear_gelu(x);
    }
    printf("somme gelu sur [-6,6] (100k evals) = %.3f\n", acc);
    return 0;
}
