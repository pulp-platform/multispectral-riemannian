/**
 * @file svd_parallel.c
 * @author Tibor Schneider
 * @date 2020/03/02
 * @brief This file contains the functions for computing the SVD in parallel
 */

/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "linalg.h"
#include "../func/functional.h"
#include "rt/rt_api.h"
#include "math.h"
#include "../insn.h"

#define _GIVENS_SAVE_MIN 1.0e-10
#define _SQRT2 1.41421353816986083984f
#define _EPSILON 1.1920928955078125e-7f // smallest e, s.t. 1 + e > 1 (from numpy)
#define _SVD_PRECISION 1e-4f
#define _MIN_ALLOWED_EIGENVALUE 1.e-3f

#ifndef NUM_WORKERS
#define NUM_WORKERS 9
#endif//NUM_WORKERS

static RT_L1_DATA linalg_evd_2x2_t _g_evd;

/*
 * Declaration of all kernel functions. Their documentation is below (where they are implemented)
 */
void _linalg_kernel_svd_sym(unsigned int core_id, float* p_a, float* p_q, unsigned int N, float* p_workspace);
void _linalg_kernel_svd_sym_tridiag(unsigned int core_id, float* p_main_diag, float* p_off_diag, float* p_q, unsigned int N, unsigned int stride, unsigned int current_pos);
void _linalg_kernel_apply_givens_rotation_f(unsigned int core_id, float* p_a, linalg_givens_rotation_t* rot, unsigned int k, unsigned int N);
void _linalg_kernel_householder_tridiagonal(unsigned int core_id, float* p_a, float* p_q, unsigned int N, float* p_workspace);
void _linalg_kernel_householder_update_step(unsigned int core_id, float* p_a, const float* p_v, const float* p_w, float c, unsigned int N, unsigned int kp1);
void _linalg_kernel_householder_update_step_Q(unsigned int core_id, float* p_q, const float* p_v, const float* p_w, unsigned int N, unsigned int kp1);

typedef struct {
    float* p_a;
    unsigned int N;
    float* p_workspace;
} _linalg_kernel_logm_instance_t;

/**
 * @brief Compute the matrix logarithm of a matrix by computing the SVD first.
 *
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the matrix logarithm of A
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (2N + 2) space
 */
void _linalg_kernel_logm(void* args) {

    _linalg_kernel_logm_instance_t* _args = args;
    float* p_a = _args->p_a;
    unsigned int N = _args->N;
    float* p_workspace = _args->p_workspace;
    unsigned int core_id = rt_core_id();

    float* _p_q = p_workspace;
    float* _p_hh_workspace = p_workspace + N * N;

    if (core_id == 0) {
        // fill _p_q with unit matrix
        linalg_fill_I(_p_q, N);
    }

    rt_team_barrier();

    // compute the householder reflection
    _linalg_kernel_householder_tridiagonal(core_id, p_a, _p_q, N, _p_hh_workspace);

    // next, generate main and off diagonal vectors
    float* _p_main_diag = p_workspace + N * N;
    float* _p_off_diag = p_workspace + N * (N + 1);

    for (unsigned int _i = core_id; _i < N - 1; _i += NUM_WORKERS) {
        _p_main_diag[_i] = p_a[_i * (N + 1)];
        _p_off_diag[_i] = p_a[_i * (N + 1) + 1];
    }
    if (core_id == NUM_WORKERS - 1) {
        _p_main_diag[N - 1] = p_a[N * N - 1];
    }

    rt_team_barrier();

    // do the svd of the tridiagonal matrix
    _linalg_kernel_svd_sym_tridiag(core_id, _p_main_diag, _p_off_diag, _p_q, N, N, 0);

    // compute the logarithm of all eigenvalues
    for (unsigned int _i = core_id; _i < N; _i += NUM_WORKERS) {
        float _val = _p_main_diag[_i];
        _val = insn_fmax(_MIN_ALLOWED_EIGENVALUE, _val); // clip the eigenvalues
        _p_main_diag[_i] = logf(_val);
    }

    rt_team_barrier();

    // reconstruct the matrix
    float* _p_construction = p_workspace + N * (N + 2);

    //TODO make sure that this is also done in parallel!
    if (core_id == 0) {
        // transpose matrix q to copy it to _p_construction
        func_copy_transpose_mat((uint32_t*)_p_q, (uint32_t*)_p_construction, N);
    }

    rt_team_barrier();

    // multiply _p_q with the log of the eigenvalues
    linalg_matmul_diag_parallel_f(core_id, NUM_WORKERS, _p_q, _p_main_diag, N);
    // do the final matrix multiplication of p_q and p_construction and store the results in p_a
    linalg_matmul_stride_parallel_f(core_id, NUM_WORKERS, _p_q, _p_construction, N, N, N, N, N, N, p_a);

}

/**
 * @brief Compute the SVD of a symmetric matrix.
 *
 * This function first computes a tridiagonalization using Householder reflection, and then repeats
 * QR decomposition steps until the matrix is diagonalized.
 *
 * @param core_id Id of the current core on the cluster
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the eigenvalues on the main diagonal.
 * @param p_q Pointer to orthogonal transformation matrix. On entering, this matrix must either be
 *            the unit matrix I or a different orthogonal matrix. After returning, this matrix
 *            contains the eigenvectors of the matrix A.
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (N + 2) space
 */
void _linalg_kernel_svd_sym(unsigned int core_id,
                            float* p_a,
                            float* p_q,
                            unsigned int N,
                            float* p_workspace) {

    // first, do householder transformation
    linalg_householder_tridiagonal(p_a, p_q, N, p_workspace);

    // next, generate main and off diagonal vectors
    float* _p_main_diag = p_workspace;
    float* _p_off_diag = p_workspace + N;

    for (unsigned int _i = 0; _i < N - 1; _i++) {
        _p_main_diag[_i] = p_a[_i * (N + 1)];
        _p_off_diag[_i] = p_a[_i * (N + 1) + 1];
    }
    _p_main_diag[N - 1] = p_a[N * N - 1];

    // do the svd of the tridiagonal matrix
    linalg_svd_sym_tridiag(_p_main_diag, _p_off_diag, p_q, N, N, 0);

    // create the diagonal matrix
    float* _p_a_iter = p_a;
    for (unsigned int _i = 0; _i < N; _i++) {
        for (unsigned int _j = 0; _j < N; _j++) {
            if (_i == _j) {
                *_p_a_iter++ = _p_main_diag[_i];
            } else {
                *_p_a_iter++ = 0.f;
            }
        }
    }

}

/**
 * @brief Compute the SVD of a symmetric tridiagonal matrix using QR decomposition:
 *
 *     A = Q D Q^T
 *
 * @param core_id Id of the current core on the cluster
 * @param p_main_diag Pointer to vector containing the diagonal elements of A, size = (N,). After
 *                    returning, this vector contains all the eigenvalues.
 * @param p_off_diag Pointer to vector containing the off-diagonal elements of A, size = (N-1, ).
 *                   This vector is deleted after returning.
 * @param p_q Pointer to Matrix of size (N, N). After returning, this vector contains the
 *            orthogonal transformation matrix as described above. On Entry, this must either
 *            contain the unit matrix I or another orthogonal transformation (e.g. from householder
 *            tridiagonalization)
 * @param N Size of matrix Q and A
 * @param stride Stride for accessing eigenvalue matrix Q (line length of original matrix Q)
 * @param current_pos Position for the current recursion step, set to 0 to start it.
 */
void _linalg_kernel_svd_sym_tridiag(unsigned int core_id,
                                    float* p_main_diag,
                                    float* p_off_diag,
                                    float* p_q,
                                    unsigned int N,
                                    unsigned int stride,
                                    unsigned int current_pos) {

    // m: main iterator
    int _m = N - 1;

    // special cases to break out of recurrence
    if (N == 1) return;

    if (N == 2) {
        if (core_id == 0) {
            // Compute the diagonalization of the 2x2 matrix
            _g_evd = linalg_evd_2x2(*p_main_diag, *p_off_diag, *(p_main_diag + 1));

            // write the eigenvalues into the main diag
            *(p_main_diag + 0) = _g_evd.ev1;

            // apply the rotation
            linalg_apply_givens_rotation_f(p_q, _g_evd.rot, current_pos, stride);

            // this needs to be here because of GCC
            *(p_main_diag + 1) = _g_evd.ev2;
        }
        rt_team_barrier();

        return;
    }

    // If N >= 3, do the normal QR decomposition
    while (_m > 0) {

        // check if the matrix left to be transformed has off diagonal elements which are zero
        for (int _k = 0; _k < _m - 1; _k++) {
            if (insn_fabs(p_off_diag[_k]) < _EPSILON) {
                // Divide and Conquer! decompose the matrix
                _linalg_kernel_svd_sym_tridiag(core_id,
                                               p_main_diag,
                                               p_off_diag,
                                               p_q,
                                               _k + 1, stride, 0);

                _linalg_kernel_svd_sym_tridiag(core_id,
                                               p_main_diag + _k + 1,
                                               p_off_diag + _k + 1,
                                               p_q,
                                               _m - _k, stride, _k + 1);

                return;
            }
        }

        // this computation must only happen on a single core

        float _x, _y;

        if (core_id == 0) {
            // do wilkinson shift
            float _shift;
            float _d = (p_main_diag[_m - 1] - p_main_diag[_m]) * 0.5f;
            if (_d == 0) {
                _shift = p_main_diag[_m] - insn_fabs(p_off_diag[_m - 1]);
            } else {
                float _off_diag_pow2 = p_off_diag[_m - 1] * p_off_diag[_m - 1];
                float _tmp = insn_fsqrt(insn_fmadd(_d, _d, _off_diag_pow2));
                _tmp = insn_fsgnj(_tmp, _d);
                _shift = p_main_diag[_m] - _off_diag_pow2 / (_d + _tmp);
            }

            // start the implicit QR step
            _x = p_main_diag[0] - _shift;
            _y = p_off_diag[0];
        }

        for (int _k = 0; _k < _m; _k++) {

            //only the first core must compute the rotation and update the main and off diagonal elements

            if (core_id == 0) {

                // determine the givens rotation
                linalg_givens_rotation_t _rot;
                if (_m > 1) {
                    _rot = linalg_givens_rotation(_x, _y);
                } else {
                    _rot = linalg_givens_rotation_diag(p_main_diag[0], p_off_diag[0], p_main_diag[1]);
                }

                // compute some values
                float _w = insn_fmsub(_rot.cs, _x, _rot.sn * _y);
                float _d = insn_fsub(p_main_diag[_k], p_main_diag[_k + 1]);
                float _z = insn_fmadd(2 * _rot.cs, p_off_diag[_k], _d * _rot.sn) * _rot.sn;

                // do the step on the main and off diagonal
                p_main_diag[_k] = insn_fsub(p_main_diag[_k], _z);
                p_main_diag[_k + 1] = insn_fadd(p_main_diag[_k + 1], _z);
                p_off_diag[_k] = insn_fmadd(p_off_diag[_k], insn_fmsub(_rot.cs, _rot.cs, _rot.sn * _rot.sn), _d * _rot.cs * _rot.sn);
                if (_k > 0) {
                    p_off_diag[_k - 1] = _w;
                }

                // update x and y
                _x = p_off_diag[_k];
                if (_k < _m - 1) {
                    _y = insn_fnmadd(_rot.sn, p_off_diag[_k + 1], 0.f);
                    p_off_diag[_k + 1] = insn_fmul(_rot.cs, p_off_diag[_k + 1]);
                }

                _rot.sn = -_rot.sn; // change the sign for the rotation because the sine is defined differently here!

                // write the rotation to the global storage
                _g_evd.rot.cs = _rot.cs;
                _g_evd.rot.sn = _rot.sn;

            }

            rt_team_barrier();

            // update the eigenvectors in parallel
            _linalg_kernel_apply_givens_rotation_f(core_id, p_q, &(_g_evd.rot), current_pos + _k, stride);

        }

        // check for convergence
        if (insn_fabs(p_off_diag[_m - 1]) < _SVD_PRECISION * (insn_fabs(p_main_diag[_m - 1]) + insn_fabs(p_main_diag[_m]))) {
            _m -= 1;
        }

    }
}

/**
 * @brief Applies the givens rotation on matrix A
 *
 * @warning this funciton operates inplace
 *
 * @param core_id Id of the current core on the cluster
 * @param p_a Pointer to matrix A of shape [N, N], will be modified
 * @param p_rot Pointer to rotation structure
 * @param k Position of the rotation, 0 <= k < N-1
 * @param N Dimension of matrix a
 */
void _linalg_kernel_apply_givens_rotation_f(unsigned int core_id,
                                            float* p_a,
                                            linalg_givens_rotation_t* p_rot,
                                            unsigned int k,
                                            unsigned int N) {

    float* _p_a_offset = p_a + k;

    float _val_a, _val_b;
    float _res_a, _res_b;

    float cs = p_rot->cs;
    float sn = p_rot->sn;

    // loop over all rows
    for (int _i = core_id; _i < N; _i += NUM_WORKERS) {

        // load the two values
        _val_a = _p_a_offset[_i * N + 0];
        _val_b = _p_a_offset[_i * N + 1];

        // compute the new values
        _res_a = insn_fmadd(_val_a, cs, _val_b * sn);
        _res_b = insn_fmsub(_val_b, cs, _val_a * sn);

        // store back
        _p_a_offset[_i * N + 0] = _res_a;
        _p_a_offset[_i * N + 1] = _res_b;
    }

    rt_team_barrier();
}

/**
 * @brief Compute the tridiagonalization of a symmetric matrix A.
 *
 *     T = Q' A  Q
 *     A = Q  T  Q'
 *
 * @param core_id Id of the current core on the cluster
 * @param p_a Pointer to symmetric matrix A. At return, will contain the trigiagonalized matrix T
 * @param p_q Pointer to orthogonal matrix Q. Input must either be the unit matrix I or any other 
 *            orthogonal matrix. At return, will contain the transformation matrix L combined with
 *            the previous matrix: Q_out = H Q_in
 * @param N Dimension of all matrices.
 * @param p_workspace Pointer to workspace, requires (N * 2) space.
 */
void _linalg_kernel_householder_tridiagonal(unsigned int core_id,
                                            float* p_a,
                                            float* p_q,
                                            unsigned int N,
                                            float* p_workspace) {

    /*
     * Fast implementation of the householder reflections
     */

    float* _p_v = p_workspace;                  // vector of size N
    float* _p_w = p_workspace + N;              // vector of size N

    float* _p_a = p_a;
    float* _p_q = p_q;

    float* _p_a_iter = _p_a;
    float* _p_v_iter;

    // Start with the iterations
    for (int _k = 0; _k < N - 2; _k++) {

        // pointer to A[k,k+1]
        _p_a_iter = _p_a + _k * (N + 1) + 1;

        if (core_id == 0) {
            // compute the scale of the row right of the current diagonal element
            float _scale = linalg_vnorm_f(_p_a_iter, (N - _k - 1), 1);
            if (_scale == 0.f) {
                continue;
            }

            float _val = *(_p_a_iter++); // _p_a_iter now points to A[k,k+2]
            float _scaled_val = insn_fabs(_val / _scale);
            float _z = (1.f + _scaled_val) * 0.5f;
            float _sqrtz = insn_fsqrt(_z);
            float _vec_scale = 1.f / (2.f * _scale * _sqrtz);

            // generate vector _p_v
            _p_v_iter = _p_v;
            for (int _i = 0; _i < _k + 1; _i++) {
                *_p_v_iter++ = 0.f;
            }
            *_p_v_iter++ = _sqrtz;
            for (int _i = _k + 2; _i < N; _i++) {
                // read the element of A and multiply with the sign of _val
                float _tmp_val = insn_fsgnjx(*_p_a_iter++, _val);
                // write the vector
                *_p_v_iter++ = _tmp_val * _vec_scale;
            }
        }

        rt_team_barrier();

        unsigned int _submat_size = N - _k - 1;

        // generate vector w
        linalg_vecmatmul_parallel_f(core_id, NUM_WORKERS, _p_v + _k + 1, _p_a + (_k + 1) * N, _submat_size, N, _p_w);

        // TODO currently, every core computes _c
        // compute constant c
        float _c = linalg_vec_innerprod_f(_p_v + _k + 1, _p_w + _k + 1, _submat_size);

        // update matrix A
        _linalg_kernel_householder_update_step(core_id, _p_a, _p_v, _p_w, _c, N, _k + 1);

        // generate w = Q v^T
        linalg_matvecmul_parallel_f(core_id, NUM_WORKERS, _p_q + _k + 1, _p_v + _k + 1, N, _submat_size, N, _p_w);

        // update Q <- Q - 2 Q v v^T = Q - 2 w v^T
        _linalg_kernel_householder_update_step_Q(core_id, _p_q, _p_v, _p_w, N, _k + 1);

    }
}

/**
 * @brief update matrix A inside the householder tridiagonalization
 *
 *     A = A - 2 (vw^T + wv^T) + 4 * c * v v^T
 *
 * @param core_id Id of the current core on the cluster
 * @param p_a Pointer to matrix A of shape [N, N], is updated in place
 * @param p_v Pointer to vector v of shape [N], all values up to k+1 are assumed to be zero
 * @param p_w Pointer to vector w of shape [N]
 * @param c constant factor c
 * @param N dimensionality
 * @param kp1 Part of the vector v which is zero (kp1 = k + 1)
 */
void _linalg_kernel_householder_update_step(unsigned int core_id,
                                            float* p_a,
                                            const float* p_v,
                                            const float* p_w,
                                            float c,
                                            unsigned int N,
                                            unsigned int kp1) {

    // We have three regions: 
    // 1. the upper left part which is not touched,
    // 2. the upper right part where only 2ddt is subtracted
    // 3. the lower right part where we subtract 2ddt and add c4 * vvt

    float _v_i_2, _w_i_2; // values of v and w at position i, multiplied by 2
    float _v_j, _w_j;     // values of v and w at position j
    float _updated_a; // new value for a at position (i, j) and (j, i)

    // Region 2

    // two different methods: if kp1 < NUM_WORKERS, or kp1 >= NUM_WORKERS

    if (kp1 < NUM_WORKERS) {
        for (int _i = 0; _i < kp1; _i++) {

            //_v_i_2 = 0.f // v_i is always 0 at these positions
            _w_i_2 = p_w[_i] * 2.f;

            for (int _j = kp1 + core_id; _j < N; _j += NUM_WORKERS) {

                _v_j = p_v[_j];
                _updated_a = p_a[_i * N + _j];

                // behavior with fmadd
                _updated_a = insn_fnmsub(_v_j, _w_i_2, _updated_a);

                // write back the value
                p_a[_i * N + _j] = _updated_a;
                p_a[_j * N + _i] = _updated_a;
            }
        }
    } else { // kp1 >= NUM_WORKERS
        for (int _i = core_id; _i < kp1; _i+= NUM_WORKERS) {

            //_v_i_2 = 0.f // v_i is always 0 at these positions
            _w_i_2 = p_w[_i] * 2.f;

            for (int _j = kp1; _j < N; _j++) {

                _v_j = p_v[_j];
                _updated_a = p_a[_i * N + _j];

                // behavior with fmadd
                _updated_a = insn_fnmsub(_v_j, _w_i_2, _updated_a);

                // write back the value
                p_a[_i * N + _j] = _updated_a;
                p_a[_j * N + _i] = _updated_a;
            }
        }
    }

    // Region 3 (diagonal part): 2ddt = 4 * v[ij] * w[ij]
    float _v_v_i_4; // v_i * v_i * 4
    // The last core will get most of the work, because it was different before
    for (int _ij = kp1 + (NUM_WORKERS - core_id - 1); _ij < N; _ij += NUM_WORKERS) {

        _w_i_2 = p_w[_ij] * 2.f;
        _v_i_2 = p_v[_ij] * 2.f;

        _updated_a = p_a[_ij * N + _ij];

        // update a with the 2ddt part
        _updated_a = insn_fnmsub(_w_i_2, _v_i_2, _updated_a);

        // update with the 4c vvt part
        _v_v_i_4 = _v_i_2 * _v_i_2;
        _updated_a = insn_fmadd(c, _v_v_i_4, _updated_a);

        // write back value
        p_a[_ij * N + _ij] = _updated_a;
    }

    // Since one of the v in 4cvvt is already multiplied by 2, we want c to multiply by 2
    float _c_2 = c * 2.f;
    float _c_v_i_4;
    float _2ddt;

    // Region 3 (Nondiagonal part)
    // triangular parallelization using a skip variable
    unsigned int _skip_step = core_id;

    for (int _i = kp1; _i < N - 1; _i++) {

        _v_i_2 = p_v[_i] * 2.f;
        _w_i_2 = p_w[_i] * 2.f;
        _c_v_i_4 = _v_i_2 * _c_2;

        for (int _j = _i + 1; _j < N; _j++) {
            if (_skip_step > 0) {
                _skip_step--;
                continue;
            }
            _skip_step = NUM_WORKERS - 1;

            _v_j = p_v[_j];
            _w_j = p_w[_j];

            _updated_a = p_a[_i * N + _j];

            // update with the 4cvvt part
            _updated_a = insn_fmadd(_c_v_i_4, _v_j, _updated_a);

            // update with the 2ddt part
            _2ddt = _v_i_2 * _w_j;
            _2ddt = insn_fmadd(_v_j, _w_i_2, _2ddt);

            _updated_a -= _2ddt;

            // write back the value
            p_a[_i * N + _j] = _updated_a;
            p_a[_j * N + _i] = _updated_a;
        }
    }

    rt_team_barrier();
}

/**
 * @brief updates matrix Q inside the householder tridiagonalization
 *
 *     Q = Q - 2 * w v^T  (both v and w are assumed to be column vectors)
 *
 * @param core_id Id of the current core on the cluster
 * @param p_q Pointer to matrix Q, of shape [N, N], is updated in place
 * @param p_v Pointer to vector v of shape [N], all values up to k+1 are assumed to be zero
 * @param p_w Pointer to vector w of shape [N]
 * @param N Dimensionality of Q and both vectors v and w
 * @param kp1 Part of the matrices which are zero (k + 1)
 */
void _linalg_kernel_householder_update_step_Q(unsigned int core_id,
                                              float* p_q,
                                              const float* p_v,
                                              const float* p_w,
                                              unsigned int N,
                                              unsigned int kp1) {

    // we have two regions, one is never used, and the other must be updated

    float _w_i_n2; // value of -2*w[i]
    float _v_j;
    float updated_q;

    for (unsigned int _i = core_id; _i < N; _i += NUM_WORKERS) {

        _w_i_n2 = p_w[_i] * -2.f;

        for (unsigned int _j = kp1; _j < N; _j++) {
            updated_q = p_q[_i * N + _j];
            _v_j = p_v[_j];
            updated_q = insn_fmadd(_w_i_n2, _v_j, updated_q);
            p_q[_i * N + _j] = updated_q;
        }
    }

    rt_team_barrier();
}

/**
 * @brief Compute the matrix logarithm of a matrix in parallel by computing the SVD first.
 *
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the matrix logarithm of A
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (3N + 2) space
 */
void linalg_logm_parallel(float* p_a,
                          unsigned int N,
                          float* p_workspace) {

    _linalg_kernel_logm_instance_t _args;
    _args.p_a = p_a;
    _args.N = N;
    _args.p_workspace = p_workspace;

    rt_team_fork(NUM_WORKERS, _linalg_kernel_logm, &_args);

}
