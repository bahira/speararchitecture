/* test_spear.c - audit de parite du corpus portable (compile+run).
 * gcc -std=c99 -Wall -Wextra -pedantic -O2 test_spear.c -o test_spear && ./test_spear
 */
#include <math.h>
#include <stdio.h>

#include "spear_activations.h"

static float sigmoid_ref(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu_ref(float x)    { return x * sigmoid_ref(x); }
static float gelu_ref(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}
static float softplus_ref(float x) { return logf(1.0f + expf(x)); }
static float phi_ref(float x) {
    return 0.5f * (1.0f + erff(x * 0.70710678118654752f));
}

int main(void) {
    float e_silu = 0.0f, e_gelu = 0.0f, e_sp = 0.0f, e_phi = 0.0f,
          e_phi_dom = 0.0f, e_tanh = 0.0f;
    int i;
    for (i = -800; i <= 800; ++i) {
        const float x = 0.01f * (float)i;
        float d;
        d = fabsf(spear_silu(x) - silu_ref(x));
        if (d > e_silu) e_silu = d;
        d = fabsf(spear_gelu(x) - gelu_ref(x));
        if (d > e_gelu) e_gelu = d;
        d = fabsf(spear_softplus(x) - softplus_ref(x));
        if (d > e_sp) e_sp = d;
        d = fabsf(spear_phi(x) - phi_ref(x));
        if (d > e_phi) e_phi = d;
        if (fabsf(x) <= 4.0f && d > e_phi_dom) e_phi_dom = d;
        if (fabsf(x) <= 1.0f) {
            const float dt = fabsf(spear_tanh(x) - tanhf(x));
            if (dt > e_tanh) e_tanh = dt;
        }
    }

    printf("Linf spear_silu       : %.3e\n", (double)e_silu);
    printf("Linf spear_gelu       : %.3e\n", (double)e_gelu);
    printf("Linf spear_softplus   : %.3e\n", (double)e_sp);
    printf("Linf spear_phi [-8,8] : %.3e\n", (double)e_phi);
    printf("Linf spear_phi [-4,4] : %.3e\n", (double)e_phi_dom);
    printf("Linf spear_tanh [-1,1]: %.3e\n", (double)e_tanh);

    /* W1 exact vs reference double precision */
    {
        static const float p[8] = {0.05f, 0.10f, 0.15f, 0.20f,
                                   0.20f, 0.15f, 0.10f, 0.05f};
        static const float q[8] = {0.02f, 0.08f, 0.15f, 0.20f,
                                   0.25f, 0.15f, 0.10f, 0.05f};
        double cp = 0.0, cq = 0.0, wref = 0.0;
        int k;
        for (k = 0; k < 8; ++k) {
            cp += p[k];
            cq += q[k];
            wref += fabs(cp - cq);
        }
        {
            const float w = spear_exact_wasserstein1(p, q, 8u);
            printf("W1 spear = %.6f | double ref = %.6f | diff = %.2e\n",
                   (double)w, wref, fabs((double)w - wref));
        }
    }

    {
        const int fail = (e_silu > 4.5e-2f) || (e_gelu > 5.5e-2f) ||
                         (e_sp > 2.5e-4f) || (e_phi > 5.0e-2f) ||
                         (e_phi_dom > 5.0e-3f) || (e_tanh > 2.0e-5f);
        printf("%s\n", fail ? "AUDIT FAIL" : "AUDIT PASS");
        return fail ? 1 : 0;
    }
}
