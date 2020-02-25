/**
 * @file svd.c
 * @author Tibor Schneider
 * @date 2020/02/21
 * @brief This file contains the functions for computing the SVD
 */

#include "linalg.h"
#include "../func/functional.h"
#include "rt/rt_api.h"
#include "math.h"
#include "../insn.h"

#define _GIVENS_SAVE_MIN 1.0e-10
#define _SQRT2 1.41421353816986083984f
#define _EPSILON 1.1920928955078125e-7f // smallest e, s.t. 1 + e > 1 (from numpy)

/**
 * @brief Compute the tridiagonalization of a symmetric matrix A.
 *
 *     T = Q' A  Q
 *     A = Q  T  Q'
 *
 * @param p_a Pointer to symmetric matrix A. At return, will contain the trigiagonalized matrix T
 * @param p_q Pointer to orthogonal matrix Q. Input must either be the unit matrix I or any other 
 *            orthogonal matrix. At return, will contain the transformation matrix L combined with
 *            the previous matrix: Q_out = H Q_in
 * @param N Dimension of all matrices.
 * @param p_workspace Pointer to workspace, requires (N * (2N + 1)) space.
 */
void linalg_householder_tridiagonal(float* p_a,
                                    float* p_q,
                                    unsigned int N,
                                    float* p_workspace) {

    float* _p_v = p_workspace;
    float* _p_h = p_workspace + N;
    float* _p_tmp = p_workspace + N * (N + 1);

    float* _p_a = p_a;
    float* _p_q = p_q;

    float* _p_a_iter = _p_a;
    float* _p_v_iter;
    float* _p_h_iter;

    float* _p_swap;

    // Start with the iterations
    for (int _k = 0; _k < N - 2; _k++) {

        // pointer to A[k,k+1]
        _p_a_iter = _p_a + _k * (N + 1) + 1;

        // compute the scale of the row right of the current diagonal element
        float _scale = linalg_vnorm_f(_p_a_iter, (N - _k - 1), 1);
        if (_scale == 0.f) {
            continue;
        }

        float _val = *(_p_a_iter++); // _p_a_iter now points to A[k,k+2]
        float _sign = insn_fsgnj(1.f, _val);
        // float _scaled_val = insn_fabs(_val / _scale);
        float _scaled_val = _sign * _val / _scale;
        float _z = (1.f + _scaled_val) * 0.5f;
        float _sqrtz = insn_fsqrt(_z);
        float _vec_scale = 1 / (2.f * _scale * _sqrtz);

        // generate vector _p_v
        // TODO optimize
        _p_v_iter = _p_v;
        for (int _i = 0; _i < _k + 1; _i++) {
            *_p_v_iter++ = 0.f;
        }
        *_p_v_iter++ = _sqrtz;
        for (int _i = _k + 2; _i < N; _i++) {
            // read the element of A and multiply with the sign of _val
            // float _tmp_val = insn_fsgnjx(*_p_a_iter++, _val);
            float _tmp_val = _sign * (*_p_a_iter++);
            // write the vector
            *_p_v_iter++ = _tmp_val * _vec_scale;
        }

        // Generate the rotation matrix H
        // TODO optimize
        linalg_vcovmat_f(_p_v, N, 0, _p_h);
        _p_h_iter = _p_h;
        for (int _i = 0; _i < N; _i++) {
            for (int _j = 0; _j < N; _j++) {
                if (_i == _j) {
                    *_p_h_iter = insn_fnmsub(*_p_h_iter, 2.f, 1.f);
                } else {
                    *_p_h_iter = -2.f * (*_p_h_iter);
                }
                _p_h_iter++;
            }
        }

        // transform the matrices with H
        linalg_matmul_f(_p_a, _p_h, N, N, N, _p_tmp);
        linalg_matmul_f(_p_h, _p_tmp, N, N, N, _p_a); // TODO this second matmul can be optimized!

        linalg_matmul_f(_p_q, _p_h, N, N, N, _p_tmp);

        // swap pointers tmp and q
        _p_swap = _p_tmp;
        _p_tmp = _p_q;
        _p_q = _p_swap;

    }

    /*
     * We did the rotation (N-2) times in total. Thus, if (N-2) 2 3 == 0, the data is already at the
     * correct position. 
     */
    float* _o_tmp = p_workspace + N * (N + 1); // reconstruct the original tmp matrix
    if (_p_q != p_q) {
        func_copy_mat((uint32_t*)_o_tmp, (uint32_t*)p_q, N, N, N, N);
    }

}

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
