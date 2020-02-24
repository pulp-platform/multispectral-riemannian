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
#define _SQRT2 1.41421353816986083984f

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
        float r = insn_fsqrt(insn_fmadd(a, a, b * b));
        res.cs = a / r;
        res.sn = -b / r;
    }
    return res;
}

/**
 * @brief computes the rotation of the EVD of a 2x2 symmetric matrix
 *
 * | cs -sn | | a  b | | cs -sn |^T = | ev1  0 |
 * | sn  cs | | b  c | | sn  cs |  =  | 0  ev2 |
 *
 * @warning The sine is defined differently than in linalg_evd_2x2
 *
 * @param a first element in the main-diagonal
 * @param b element in the off-diagonal
 * @param c second element in the main-diagonal
 * @returns linalg_evd_2x2_t structure
 */
linalg_givens_rotation_t linalg_givens_rotation_diag(float a,
                                                     float b,
                                                     float c) {

    float sm = a + c;
    float df = a - c;
    float tb = b + b;
    float adf = fabs(df);
    float ab = fabs(tb);

    float rt;
    int sgn1;
    int sgn2;
    float cs;
    float ct;
    float tn;

    linalg_givens_rotation_t res;

    if (adf > ab) {
        rt = adf * insn_fsqrt(1.f + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * insn_fsqrt(1.f + (adf / ab) * (adf / ab));
    } else {
        rt = ab * _SQRT2;
    }

    if (sm < 0.f) {
        sgn1 = -1;
    } else {
        sgn1 = 1;
    }

    if (df >= 0.f) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }

    if (fabs(cs) > ab) {
        ct = -tb / cs;
        res.sn = 1.f / insn_fsqrt(1.f + ct * ct);
        res.cs = res.sn * ct;
    } else {
        if (ab == 0.f) {
            res.cs = 1.f;
            res.sn = 0.f;
        } else {
            tn = -cs / tb;
            res.cs = 1.f / insn_fsqrt(1.f + tn * tn);
            res.sn = tn * res.cs;
        }
    }

    if (sgn1 == sgn2) {
        tn = res.cs;
        res.cs = -res.sn;
        res.sn = -tn;
    } else {
        res.sn = -res.sn;
    }

    return res;
}

/**
 * @brief Computes the eigenvalue decomposition of a 2x2 symmetri matrix (lapack: SLAEV2)
 *
 * | cs -sn |^T | a  b | | cs -sn | = | ev1  0 |
 * | sn  cs |   | b  c | | sn  cs | = | 0  ev2 |
 *
 * @param a first element in the main-diagonal
 * @param b element in the off-diagonal
 * @param c second element in the main-diagonal
 * @returns linalg_evd_2x2_t structure
 */
linalg_evd_2x2_t linalg_evd_2x2(float a,
                                float b,
                                float c) {

    float sm = a + c;
    float df = a - c;
    float tb = b + b;
    float adf = fabs(df);
    float ab = fabs(tb);

    float acmx;
    float acmn;
    float rt;
    int sgn1;
    int sgn2;
    float cs;
    float ct;
    float tn;

    linalg_evd_2x2_t res;

    if (fabs(a) > fabs(c)) {
        acmx = a;
        acmn = c;
    } else {
        acmx = c;
        acmn = a;
    }

    if (adf > ab) {
        rt = adf * insn_fsqrt(1.f + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * insn_fsqrt(1.f + (adf / ab) * (adf / ab));
    } else {
        rt = ab * _SQRT2;
    }

    if (sm < 0.f) {
        res.ev1 = 0.5f * (sm - rt);
        sgn1 = -1;
        res.ev2 = (acmx / res.ev1) * acmn - (b / res.ev1) * b;
    } else if (sm > 0.f) {
        res.ev1 = 0.5f * (sm + rt);
        sgn1 = 1;
        res.ev2 = (acmx / res.ev1) * acmn - (b / res.ev1) * b;
    } else {
        res.ev1 = 0.5f * rt;
        res.ev2 = -0.5f * rt;
        sgn1 = 1;
    }

    if (df >= 0.f) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }

    if (fabs(cs) > ab) {
        ct = -tb / cs;
        res.sn = 1.f / insn_fsqrt(1.f + ct * ct);
        res.cs = res.sn * ct;
    } else {
        if (ab == 0.f) {
            res.cs = 1.f;
            res.sn = 0.f;
        } else {
            tn = -cs / tb;
            res.cs = 1.f / insn_fsqrt(1.f + tn * tn);
            res.sn = tn * res.cs;
        }
    }

    if (sgn1 == sgn2) {
        tn = res.cs;
        res.cs = -res.sn;
        res.sn = tn;
    }

    return res;
}
