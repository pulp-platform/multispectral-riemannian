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
#define _SVD_PRECISION 1e-4f
#define _MIN_ALLOWED_EIGENVALUE 1.e-3f

/**
 * @brief Compute the matrix logarithm of a matrix by computing the SVD first.
 *
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the matrix logarithm of A
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (3N + 2) space
 */
void linalg_logm(float* p_a,
                 unsigned int N,
                 float* p_workspace) {

    float* _p_q = p_workspace;
    float* _p_hh_workspace = p_workspace + N * N;

    // fill _p_q with unit matrix
    linalg_fill_I(_p_q, N);

    // compute the householder reflection
    linalg_householder_tridiagonal(p_a, _p_q, N, _p_hh_workspace);

    // next, generate main and off diagonal vectors
    float* _p_main_diag = p_workspace + N * N;
    float* _p_off_diag = p_workspace + N * (N + 1);

    for (unsigned int _i = 0; _i < N - 1; _i++) {
        _p_main_diag[_i] = p_a[_i * (N + 1)];
        _p_off_diag[_i] = p_a[_i * (N + 1) + 1];
    }
    _p_main_diag[N - 1] = p_a[N * N - 1];

    // do the svd of the tridiagonal matrix
    linalg_svd_sym_tridiag(_p_main_diag, _p_off_diag, _p_q, N, N, 0);

    // compute the logarithm of all eigenvalues
    for (unsigned int _i = 0; _i < N; _i++) {
        float _val = _p_main_diag[_i];
        _val = insn_fmax(_MIN_ALLOWED_EIGENVALUE, _val); // clip the eigenvalues
        _p_main_diag[_i] = logf(_val);
    }

    // reconstruct the matrix
    float* _p_construction = p_workspace + N * (N + 2);
    // transpose matrix q to copy it to _p_construction
    func_copy_transpose_mat((uint32_t*)_p_q, (uint32_t*)_p_construction, N);
    // multiply _p_q with the log of the eigenvalues
    linalg_matmul_diag_f(_p_q, _p_main_diag, N);
    // do the final matrix multiplication of p_q and p_construction and store the results in p_a
    linalg_matmul_f(_p_q, _p_construction, N, N, N, p_a);

}

/**
 * @brief Compute the SVD of a symmetric matrix.
 *
 * This function first computes a tridiagonalization using Householder reflection, and then repeats
 * QR decomposition steps until the matrix is diagonalized.
 *
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the eigenvalues on the main diagonal.
 * @param p_q Pointer to orthogonal transformation matrix. On entering, this matrix must either be
 *            the unit matrix I or a different orthogonal matrix. After returning, this matrix
 *            contains the eigenvectors of the matrix A.
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (2N + 2) space
 */
void linalg_svd_sym(float* p_a,
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
void linalg_svd_sym_tridiag(float* p_main_diag,
                            float* p_off_diag,
                            float* p_q,
                            unsigned int N,
                            unsigned int stride,
                            unsigned int current_pos) {

    // m: main iterator
    int _m = N - 1;

    // special cases to break out of recurrence
    if (N == 1) {
        return;
    }

    if (N == 2) {
        // Compute the diagonalization of the 2x2 matrix
        linalg_evd_2x2_t _evd = linalg_evd_2x2(*p_main_diag, *p_off_diag, *(p_main_diag + 1));

        // write the eigenvalues into the main diag
        *(p_main_diag + 0) = _evd.ev1;

        // apply the rotation
        linalg_givens_rotation_t _rot = {_evd.cs, _evd.sn};
        linalg_apply_givens_rotation_f(p_q, _rot, current_pos, stride);

        // this instruction needs to be here, because GCC fails if the two instructions are one after the other
        // If GCC does not like the code like this, fall back to O2
        *(p_main_diag + 1) = _evd.ev2;

        return;
    }

    // If N >= 3, do the normal QR decomposition
    while (_m > 0) {

        // check if the matrix left to be transformed has off diagonal elements which are zero
        for (int _k = 0; _k < _m - 1; _k++) {
            if (insn_fabs(p_off_diag[_k]) < _EPSILON) {
                // Divide and Conquer! decompose the matrix
                linalg_svd_sym_tridiag(p_main_diag,
                                       p_off_diag,
                                       p_q,
                                       _k + 1, stride, 0);

                linalg_svd_sym_tridiag(p_main_diag + _k + 1,
                                       p_off_diag + _k + 1,
                                       p_q,
                                       _m - _k, stride, _k + 1);

                // return
                return;
            }
        }

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
        float _x = p_main_diag[0] - _shift;
        float _y = p_off_diag[0];

        for (int _k = 0; _k < _m; _k++) {

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

            // update the eigenvectors
            _rot.sn = -_rot.sn; // change the sign for the rotation because the sine is defined differently here!
            linalg_apply_givens_rotation_f(p_q, _rot, current_pos + _k, stride);

        }

        // check for convergence
        if (insn_fabs(p_off_diag[_m - 1]) < _SVD_PRECISION * (insn_fabs(p_main_diag[_m - 1]) + insn_fabs(p_main_diag[_m]))) {
            _m -= 1;
        }

    }

    return;

}

/**
 * @brief Applies the givens rotation on matrix A
 *
 * @warning this funciton operates inplace
 *
 * @param p_a Pointer to matrix A of shape [N, N], will be modified
 * @param rot Rotation structure
 * @param k Position of the rotation, 0 <= k < N-1
 * @param N Dimension of matrix a
 */
void linalg_apply_givens_rotation_f(float* p_a,
                                    linalg_givens_rotation_t rot,
                                    unsigned int k,
                                    unsigned int N) {

    float* _p_a_iter = p_a + k;

    float _val_a, _val_b;
    float _res_a, _res_b;

    // loop over all rows
    for (int _i = 0; _i < N; _i++) {
        // load the two values
        _val_a = *(_p_a_iter + 0);
        _val_b = *(_p_a_iter + 1);

        // compute the new values
        _res_a = insn_fmadd(_val_a, rot.cs, _val_b * rot.sn);
        _res_b = insn_fmsub(_val_b, rot.cs, _val_a * rot.sn);

        // store back
        *(_p_a_iter + 0) = _res_a;
        *(_p_a_iter + 1) = _res_b;

        // go to the next line
        _p_a_iter += N;
    }
}

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
 * @param p_workspace Pointer to workspace, requires (N * (2N + 2)) space.
 */
void linalg_householder_tridiagonal(float* p_a,
                                    float* p_q,
                                    unsigned int N,
                                    float* p_workspace) {

#define HOUSEHOLDER_FAST
#ifdef HOUSEHOLDER_FAST

    /*
     * Fast implementation of the householder reflections
     */

    float* _p_v = p_workspace;                  // vector of size N
    float* _p_w = p_workspace + N;              // vector of size N
    float* _p_d = p_workspace + 2 * N;          // matrix of size N * N
    float* _p_vv = p_workspace + 2 * N + N * N; // matrix of size N * N

    float* _p_a = p_a;
    float* _p_q = p_q;

    float* _p_a_iter = _p_a;
    float* _p_v_iter;

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
        float _scaled_val = insn_fabs(_val / _scale);
        float _z = (1.f + _scaled_val) * 0.5f;
        float _sqrtz = insn_fsqrt(_z);
        float _vec_scale = 1.f / (2.f * _scale * _sqrtz);

        // generate vector _p_v
        // TODO optimize (but it actually does not matter, all time is spent in matmul)
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

        unsigned int _submat_size = N - _k - 1;

        // generate vector w
        linalg_vecmatmul_f(_p_v + _k + 1, _p_a + (_k + 1) * N, _submat_size, N, _p_w);

        // compute constant c
        float _c = linalg_vec_innerprod_f(_p_v + _k + 1, _p_w + _k + 1, _submat_size);

        // update matrix A
        linalg_householder_update_step(_p_a, _p_v, _p_w, _c, N, _k + 1);

        // generate w = Q v^T
        linalg_matvecmul_f(_p_q + _k + 1, _p_v + _k + 1, N, _submat_size, N, _p_w);

        // generate _p_vv = v w
        linalg_vec_outerprod_f(_p_w, _p_v + _k + 1, N, _submat_size, N, _p_vv + _k + 1);

        // update Q <- Q - 2 Q v v^T = Q - 2 w v^T
        linalg_householder_update_step_Q(_p_q, _p_vv, N, _k + 1);

    }

#else //HOUSEHOLDER_FAST

    /*
     * Slow implementation of the householder reflections
     */

    //workspace use for slow implementation
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
        float _scaled_val = insn_fabs(_val / _scale);
        float _z = (1.f + _scaled_val) * 0.5f;
        float _sqrtz = insn_fsqrt(_z);
        float _vec_scale = 1.f / (2.f * _scale * _sqrtz);

        // generate vector _p_v
        // TODO optimize (but it actually does not matter, all time is spent in matmul)
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

        // Generate the rotation matrix H
        // TODO optimize (but it actually does not matter, all time is spent in matmul)
        linalg_vcovmat_f(_p_v, N, N, 0, _p_h);
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

        /*
         * Slow householder implementation
         */

        // transform the matrices with H
        unsigned int _submat_size = N - _k - 1;

        //linalg_matmul_f(_p_a, _p_h, N, N, N, _p_tmp);
        func_copy_mat((uint32_t*)_p_a, (uint32_t*)_p_tmp, N, _k + 1, N, N); // TODO only copy the necessary part
        linalg_matmul_stride_f(_p_a + _k + 1,
                               _p_h + (_k + 1) * (N + 1),
                               N, _submat_size, _submat_size,
                               N, N, N,
                               _p_tmp + _k + 1);

        //linalg_matmul_to_sym_f(_p_h, _p_tmp, N, _p_a);
        //linalg_matmul_f(_p_h, _p_tmp, N, N, N, _p_a);
        func_copy_mat((uint32_t*)_p_tmp, (uint32_t*)_p_a, _k + 1, N, N, N);
        linalg_matmul_stride_f(_p_h + (_k + 1) * (N + 1),
                               _p_tmp + (_k + 1) * N,
                               _submat_size, _submat_size, N,
                               N, N, N,
                               _p_a + (_k + 1) * N);

        //linalg_matmul_f(_p_q, _p_h, N, N, N, _p_tmp);
        func_copy_mat((uint32_t*)_p_q, (uint32_t*)_p_tmp, N, _k + 1, N, N);
        linalg_matmul_stride_f(_p_q + _k + 1,
                               _p_h + (_k + 1) * (N + 1),
                               N, _submat_size, _submat_size,
                               N, N, N,
                               _p_tmp + _k + 1);

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

#endif //HOUSEHOLDER_FAST

}

/**
 * @brief update matrix A inside the householder tridiagonalization
 *
 *     A = A - 2 (vw^T + wv^T) + 4 * c * v v^T
 *
 * @param p_a Pointer to matrix A of shape [N, N], is updated in place
 * @param p_v Pointer to vector v of shape [N], all values up to k+1 are assumed to be zero
 * @param p_w Pointer to vector w of shape [N]
 * @param c constant factor c
 * @param N dimensionality
 * @param kp1 Part of the vector v which is zero (kp1 = k + 1)
 */
void linalg_householder_update_step(float* p_a,
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

    for (int _i = 0; _i < kp1; _i++) {

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

    // Region 3 (diagonal part): 2ddt = 4 * v[ij] * w[ij]
    float _v_v_i_4; // v_i * v_i * 4
    for (int _ij = kp1; _ij < N; _ij++) {

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

    for (int _i = kp1; _i < N - 1; _i++) {

        _v_i_2 = p_v[_i] * 2.f;
        _w_i_2 = p_w[_i] * 2.f;
        _c_v_i_4 = _v_i_2 * _c_2;

        for (int _j = _i + 1; _j < N; _j++) {
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
}

/**
 * @brief updates matrix Q inside the householder tridiagonalization
 *
 *     Q = Q - 2 * vvt
 *
 * @param p_q Pointer to matrix Q, of shape [N, N], is updated in place
 * @param p_vvt Pointer to matrix v v^T, only the nonzero right part is used
 * @param N Dimensionality of A, 2ddt and vvt
 * @param kp1 Part of the matrices which are zero (k + 1)
 */
void linalg_householder_update_step_Q(float* p_q,
                                      const float* p_vvt,
                                      unsigned int N,
                                      unsigned int kp1) {

    float _val_q, _val_vvt;

    /*
     * We have two regions, one is never used, and the other must be updated
     */

    for (int _i = 0; _i < N; _i++) {
        for (int _j = kp1; _j < N; _j++) {
            _val_q = p_q[_i * N + _j];
            _val_vvt = p_vvt[_i * N + _j];

            p_q[_i * N + _j] = insn_fnmsub(_val_vvt, 2.f, _val_q);
        }
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
        rt = adf * insn_fsqrt(insn_fmadd((ab / adf), (ab / adf), 1.f));
    } else if (adf < ab) {
        rt = ab * insn_fsqrt(insn_fmadd((adf / ab), (adf / ab), 1.f));
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
        res.sn = 1.f / insn_fsqrt(insn_fmadd(ct, ct, 1.f));
        res.cs = res.sn * ct;
    } else {
        if (ab == 0.f) {
            res.cs = 1.f;
            res.sn = 0.f;
        } else {
            tn = -cs / tb;
            res.cs = 1.f / insn_fsqrt(insn_fmadd(tn ,tn, 1.f));
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
        rt = adf * insn_fsqrt(insn_fmadd((ab / adf), (ab / adf), 1.f));
    } else if (adf < ab) {
        rt = ab * insn_fsqrt(insn_fmadd((adf / ab), (adf / ab), 1.f));
    } else {
        rt = ab * _SQRT2;
    }

    if (sm < 0.f) {
        res.ev1 = 0.5f * (sm - rt);
        sgn1 = -1;
        res.ev2 = insn_fmsub((acmx / res.ev1), acmn, (b / res.ev1) * b);
    } else if (sm > 0.f) {
        res.ev1 = 0.5f * (sm + rt);
        sgn1 = 1;
        res.ev2 = insn_fmsub((acmx / res.ev1), acmn, (b / res.ev1) * b);
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
        res.sn = 1.f / insn_fsqrt(insn_fmadd(ct, ct, 1.f));
        res.cs = res.sn * ct;
    } else {
        if (ab == 0.f) {
            res.cs = 1.f;
            res.sn = 0.f;
        } else {
            tn = -cs / tb;
            res.cs = 1.f / insn_fsqrt(insn_fmadd(tn ,tn, 1.f));
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
