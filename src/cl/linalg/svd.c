/**
 * @file svd.c
 * @author Tibor Schneider
 * @date 2020/02/21
 * @brief This file contains the functions for computing the SVD
 */

#include "linalg.h"
#include "rt/rt_api.h"
#include "math.h"
#include "../insn.h"

#define _GIVENS_SAVE_MIN 1.0e-10

/**
 * @brief Computes the givens rotation coefficients cosine and sine
 *
 * | c -s | | a | = | r |
 * | s  c | | b | = | 0 |
 *
 * @param a first value
 * @param b second value
 * @returns linalg_givens_rotation_t structure
 */
linalg_givens_rotation_t linalg_givens_rotation(float a,
                                                float b) {
    linalg_givens_rotation_t res;

    if (b == 0.f) {
        res.cs = 1.f;
        res.sn = 0.f;
    } else if (a == 0.f) {
        res.cs = 0.f;
        res.sn = copysignf(1.f, b);
    } else {
        float scale = insn_fmax(fabs(a), fabs(b));
        if (scale < _GIVENS_SAVE_MIN) {
            a = a / scale;
            b = b / scale;
        }
        float r = insn_fsqrt(a * a + b * b);
        res.cs = a / r;
        res.sn = -b / r;
    }
    return res;
}
