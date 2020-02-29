/**
 * @file matop_f.c
 * @author Tibor Schneider
 * @date 2020/02/22
 * @brief This file contains the functions for floating point matrix multiplication
 */

#include "rt/rt_api.h"
#include "linalg.h"
#include "../insn.h"

#ifdef USE_SOFT_DIVSQRT
#include "math.h"
#endif //USE_SORT_DIVSQRT

/**
 * @brief fills matrix A with the unit matrix, with one on the main diagonal.
 *
 * @param p_a: pointer to matrix A of shape (dim, dim). Must be allocated
 * @param dim: dimension of matrix A
 */
void linalg_fill_I(float* p_a, unsigned int dim) {

    float* _p_a_iter = p_a;

    int _val;

    // set diagonal values
    for (int _i = 0; _i < dim; _i++) {
        *_p_a_iter = 1.f;
        _p_a_iter += dim + 1;
    }

    // set zero values
    for (int _i = 1; _i < dim; _i++) {
        for (int _j = 0; _j < _i; _j++) {
            p_a[_i + dim * _j] = 0.f;
            p_a[_j + dim * _i] = 0.f;
        }
    }

}

/**
 * @brief Compute the matrix addition of two floating point matrices
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to matrix B of shape [M, N]
 * @param M Rows of matrix A and B
 * @param N Columns of matrix A and B
 * @param p_y Pointer to matrix Y = A + B of shape [M, N]
 */
void linalg_matadd_f(const float* p_a,
                     const float* p_b,
                     unsigned int M,
                     unsigned int N,
                     float* p_y) {
    
    const float* _p_a_iter = p_a;
    const float* _p_b_iter = p_b;
    float* _p_y_iter = p_y;

    for (int _m = 0; _m < M; _m++) {
        for (int _n = 0; _n < N; _n++) {
            *_p_y_iter++ = (*_p_a_iter++) + (*_p_b_iter++);
        }
    }
}

/**
 * @brief Compute the matrix subtraction of two floating point matrices
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to matrix B of shape [M, N]
 * @param M Rows of matrix A and B
 * @param N Columns of matrix A and B
 * @param p_y Pointer to matrix Y = A - B of shape [M, N]
 */
void linalg_matsub_f(const float* p_a,
                     const float* p_b,
                     unsigned int M,
                     unsigned int N,
                     float* p_y) {
    
    const float* _p_a_iter = p_a;
    const float* _p_b_iter = p_b;
    float* _p_y_iter = p_y;

    for (int _m = 0; _m < M; _m++) {
        for (int _n = 0; _n < N; _n++) {
            *_p_y_iter++ = (*_p_a_iter++) - (*_p_b_iter++);
        }
    }
}

/**
 * @brief Compute the matrix multiplication of two floating point matrices
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to matrix B of shape [N, O]
 * @param M Rows of matrix A and Y
 * @param N Rows of matrix B and columns of matrix A
 * @param O Columns of matrix B and Y
 * @param stride_a number of elements between the beginning of each row of matrix A, stride_a >= N
 * @param stride_b number of elements between the beginning of each row of matrix B, stride_b >= O
 * @param stride_y number of elements between the beginning of each row of matrix Y, stride_y >= O
 * @param p_y Pointer to matrix Y = AB of shape [M, O]
 */
void linalg_matmul_stride_f(const float* p_a,
                            const float* p_b,
                            unsigned int M,
                            unsigned int N,
                            unsigned int O,
                            unsigned int stride_a,
                            unsigned int stride_b,
                            unsigned int stride_y,
                            float* p_y) {

    /*
     * Compute N elements of the output matrix at a time, reducing the number of required elements 
     * to load
     */

    // use iterators
    const float* _p_a_iter = p_a;
    const float* _p_b_iter = p_b;
    const float* _p_a_iter_comp;
    const float* _p_b_iter_comp;
    float* _p_y_iter = p_y;

    float _acc0;
    float _acc1;
    float _acc2;
    float _acc3;

    float _val_a;

    unsigned int _num_block = O / 4;
    unsigned int _rem_block = O % 4;

    // loop over every output element
    for (unsigned int _m = 0; _m < M; _m++) {
        for (unsigned int _o_blk = 0; _o_blk < _num_block; _o_blk++) {

            _acc0 = 0;
            _acc1 = 0;
            _acc2 = 0;
            _acc3 = 0;
            _p_a_iter_comp = _p_a_iter;
            _p_b_iter_comp = _p_b_iter;

            for (unsigned int _n = 0; _n < N; _n++) {
                _val_a = *_p_a_iter_comp;

                _acc0 = insn_fmadd(_val_a, *(_p_b_iter_comp + 0), _acc0);
                _acc1 = insn_fmadd(_val_a, *(_p_b_iter_comp + 1), _acc1);
                _acc2 = insn_fmadd(_val_a, *(_p_b_iter_comp + 2), _acc2);
                _acc3 = insn_fmadd(_val_a, *(_p_b_iter_comp + 3), _acc3);

                _p_a_iter_comp++;
                _p_b_iter_comp += stride_b;
            }

            *(_p_y_iter++) = _acc0;
            *(_p_y_iter++) = _acc1;
            *(_p_y_iter++) = _acc2;
            *(_p_y_iter++) = _acc3;

            _p_b_iter += 4;
        }

        // compute the remaining elements
        for (unsigned int _o = 0; _o < _rem_block; _o++) {

            _acc0 = 0;
            _p_a_iter_comp = _p_a_iter;
            _p_b_iter_comp = _p_b_iter;

            for (unsigned int _n = 0; _n < N; _n++) {

                _acc0 = insn_fmadd(*_p_a_iter_comp, *_p_b_iter_comp, _acc0);

                _p_a_iter_comp++;
                _p_b_iter_comp += stride_b;
            }

            *(_p_y_iter++) = _acc0;
            _p_b_iter++;

        }
        _p_a_iter += stride_a;
        _p_b_iter = p_b;
        _p_y_iter += (stride_y - O);
    }

}


/**
 * @brief Compute the matrix multiplication of two floating point matrices, where the result
 * is already known to be symmetric (used in Householder Transformation)
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [N, N]
 * @param p_b Pointer to matrix B of shape [N, N]
 * @param N Rows and columns of all matrices
 * @param p_y Pointer to matrix Y = AB of shape [N, N]
 */
void linalg_matmul_to_sym_f(const float* p_a,
                            const float* p_b,
                            unsigned int N,
                            float* p_y) {

    // use iterators
    const float* _p_a_iter_comp;
    const float* _p_b_iter_comp;

    float _acc0, _acc1, _acc2, _acc3;
    float _val;

    // loop over every output element
    for (unsigned int _m = 0; _m < N; _m++) {

        unsigned int num_block = _m % 4;
        unsigned int rem_block = _m % 4;

        unsigned int _o;

        for (_o = 0; _o <= num_block; _o++) {

            _acc0 = 0;
            _acc1 = 0;
            _acc2 = 0;
            _acc3 = 0;

            _p_a_iter_comp = p_a + N * _m;
            _p_b_iter_comp = p_b + _o * 4;

            for (unsigned int _n = 0; _n < N; _n++) {
                _val = *_p_a_iter_comp;
                _acc0 = insn_fmadd(_val, *(_p_b_iter_comp + 0), _acc0);
                _acc1 = insn_fmadd(_val, *(_p_b_iter_comp + 1), _acc1);
                _acc2 = insn_fmadd(_val, *(_p_b_iter_comp + 2), _acc2);
                _acc3 = insn_fmadd(_val, *(_p_b_iter_comp + 3), _acc3);

                _p_a_iter_comp++;
                _p_b_iter_comp += N;
            }

            p_y[_m * N + (_o * 4 + 0)] = _acc0;
            p_y[(_o * 4 + 0) * N + _m] = _acc0;

            p_y[_m * N + (_o * 4 + 1)] = _acc1;
            p_y[(_o * 4 + 1) * N + _m] = _acc1;

            p_y[_m * N + (_o * 4 + 2)] = _acc2;
            p_y[(_o * 4 + 2) * N + _m] = _acc2;

            p_y[_m * N + (_o * 4 + 3)] = _acc3;
            p_y[(_o * 4 + 3) * N + _m] = _acc3;
        }

        for (_o = _o * 4; _o <= _m; _o++) {

            _acc0 = 0;

            _p_a_iter_comp = p_a + N * _m;
            _p_b_iter_comp = p_b + _o;

            for (unsigned int _n = 0; _n < N; _n++) {
                _val = *_p_a_iter_comp;
                _acc0 = insn_fmadd(_val, *(_p_b_iter_comp + 0), _acc0);

                _p_a_iter_comp++;
                _p_b_iter_comp += N;
            }

            p_y[_m * N + (_o + 0)] = _acc0;
            p_y[(_o + 0) * N + _m] = _acc0;
        }
    }

}

/**
 * @brief Compute the matrix multiplication of two floating point matrices, where the second one
 * is transposed
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to matrix B of shape [O, N]
 * @param M Rows of matrix A and Y
 * @param N Columns of matrix A and B
 * @param O Rows of matrix B and columns of matrix Y
 * @param p_y Pointer to matrix Y = AB^T of shape [M, O]
 */
void linalg_matmul_T_f(const float* p_a,
                       const float* p_b,
                       unsigned int M,
                       unsigned int N,
                       unsigned int O,
                       float* p_y) {

    /*
     * Compute N elements of the output matrix at a time, reducing the number of required elements 
     * to load
     */

    // use iterators
    const float* _p_a_iter = p_a;
    const float* _p_b_iter = p_b;
    const float* _p_a_iter_comp;
    const float* _p_b_iter_comp;
    float* _p_y_iter = p_y;

    float _acc0;
    float _acc1;
    float _acc2;
    float _acc3;

    float _val_a;

    unsigned int _num_block = O / 4;
    unsigned int _rem_block = O % 4;

    // loop over every output element
    for (unsigned int _m = 0; _m < M; _m++) {
        for (unsigned int _o_blk = 0; _o_blk < _num_block; _o_blk++) {

            _acc0 = 0;
            _acc1 = 0;
            _acc2 = 0;
            _acc3 = 0;
            _p_a_iter_comp = _p_a_iter;
            _p_b_iter_comp = _p_b_iter;

            for (unsigned int _n = 0; _n < N; _n++) {
                _val_a = *_p_a_iter_comp;
                _acc0 = insn_fmadd(_val_a, *(_p_b_iter_comp + 0 * N), _acc0);
                _acc1 = insn_fmadd(_val_a, *(_p_b_iter_comp + 1 * N), _acc1);
                _acc2 = insn_fmadd(_val_a, *(_p_b_iter_comp + 2 * N), _acc2);
                _acc3 = insn_fmadd(_val_a, *(_p_b_iter_comp + 3 * N), _acc3);

                _p_a_iter_comp++;
                _p_b_iter_comp++;
            }

            *(_p_y_iter++) = _acc0;
            *(_p_y_iter++) = _acc1;
            *(_p_y_iter++) = _acc2;
            *(_p_y_iter++) = _acc3;

            _p_b_iter += 4 * N;
        }

        // compute the remaining elements
        for (unsigned int _o = 0; _o < _rem_block; _o++) {

            _acc0 = 0;
            _p_a_iter_comp = _p_a_iter;
            _p_b_iter_comp = _p_b_iter;

            for (unsigned int _n = 0; _n < N; _n++) {

                _acc0 = insn_fmadd(*_p_a_iter_comp, *_p_b_iter_comp, _acc0);

                _p_a_iter_comp++;
                _p_b_iter_comp++;
            }

            *(_p_y_iter++) = _acc0;
            _p_b_iter += N;

        }
        _p_a_iter += N;
        _p_b_iter = p_b;
    }

}

/**
 * @brief computes the matrix multiplication of a matrix A and a diagonal matrix D.
 *
 *     A <-- A @ D
 *
 * @warning The matrix A will be overwritten with the result
 *
 * @param p_a Pointer to matrix A of shape [N, N], will be overwritten
 * @param p_diag Pointer to diagonal vector of matrix D, of shape [N]
 * @param N Dimension of the matrix A and length of diagonal vector D
 */
void linalg_matmul_diag_f(float* p_a,
                          const float* p_diag,
                          unsigned int N) {

    float* _p_a_iter = p_a;
    const float* _p_diag_iter = p_diag;

    float _diag;
    float _val0, _val1, _val2, _val3;

    unsigned int num_blk = N / 4;
    unsigned int rem_blk = N % 4;

    for (unsigned int _i = 0; _i < N; _i++) {

        // load the current diagonal element
        _diag = *_p_diag_iter++;
        _p_a_iter = p_a + _i;

        for (unsigned int _j = 0; _j < num_blk; _j++) {
            _val0 = *(_p_a_iter + 0 * N);
            _val1 = *(_p_a_iter + 1 * N);
            _val2 = *(_p_a_iter + 2 * N);
            _val3 = *(_p_a_iter + 3 * N);

            _val0 = _diag * _val0;
            _val1 = _diag * _val1;
            _val2 = _diag * _val2;
            _val3 = _diag * _val3;

            *(_p_a_iter + 0 * N) = _val0;
            *(_p_a_iter + 1 * N) = _val1;
            *(_p_a_iter + 2 * N) = _val2;
            *(_p_a_iter + 3 * N) = _val3;

            _p_a_iter += 4 * N;
        }

        for (unsigned int _j = 0; _j < rem_blk; _j++) {
            _val0 = *_p_a_iter;
            _val0 = _diag * _val0;
            *_p_a_iter = _val0;
            _p_a_iter += N;
        }

    }

}

/**
 * @brief Compute the matrix vector multiplication. b is assumed to be a column vector
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to vector b of shape [N]
 * @param M Rows of matrix A and length of vector y
 * @param N columns of matrix A and length of vector b
 * @param stride_a number of elements between the beginning of each row of matrix A, stride_a >= N
 * @param p_y Pointer to vector y = Ab of shape [M]
 */
void linalg_matvecmul_f(const float* p_a,
                        const float* p_b,
                        unsigned int M,
                        unsigned int N,
                        unsigned int stride_a,
                        float* p_y) {

    const float* _p_a_iter;
    const float* _p_b_iter;
    float* _p_y_iter = p_y;

    float _acc0, _acc1, _acc2, _acc3;
    float _val_a0, _val_a1, _val_a2, _val_a3;
    float _val_b;

    unsigned int _num_blk = M / 4;
    unsigned int _rem_blk = M % 4;

    for (int _m = 0; _m < _num_blk; _m++) {

        _acc0 = 0.f;
        _acc1 = 0.f;
        _acc2 = 0.f;
        _acc3 = 0.f;
        _p_a_iter = p_a + 4 * _m * stride_a;
        _p_b_iter = p_b;

        for (int _n = 0; _n < N; _n++) {
            _val_b = *_p_b_iter++;
            _val_a0 = *(_p_a_iter + 0 * stride_a);
            _val_a1 = *(_p_a_iter + 1 * stride_a);
            _val_a2 = *(_p_a_iter + 2 * stride_a);
            _val_a3 = *(_p_a_iter + 3 * stride_a);

            _p_a_iter++;

            _acc0 = insn_fmadd(_val_b, _val_a0, _acc0);
            _acc1 = insn_fmadd(_val_b, _val_a1, _acc1);
            _acc2 = insn_fmadd(_val_b, _val_a2, _acc2);
            _acc3 = insn_fmadd(_val_b, _val_a3, _acc3);
        }

        *_p_y_iter++ = _acc0;
        *_p_y_iter++ = _acc1;
        *_p_y_iter++ = _acc2;
        *_p_y_iter++ = _acc3;

    }

    for (int _m = 0; _m < _rem_blk; _m++) {

        _acc0 = 0.f;
        _p_a_iter = p_a + (4 * _num_blk + _m) * stride_a;
        _p_b_iter = p_b;

        for (int _n = 0; _n < N; _n++) {
            _val_b = *_p_b_iter++;
            _val_a0 = *(_p_a_iter);

            _p_a_iter++;

            _acc0 = insn_fmadd(_val_b, _val_a0, _acc0);
        }

        *_p_y_iter++ = _acc0;

    }

}

/**
 * @brief Compute the vector matrix multiplication. Vector a is assumed to be a row vector
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to vector a of shape [M]
 * @param p_b Pointer to matrix B of shape [M, N]
 * @param M length of vector a and columns of matrix B
 * @param N rows of matrix B and length of vector y
 * @param p_y Pointer to vector y = Ab of shape [N]
 */
void linalg_vecmatmul_f(const float* p_a,
                        const float* p_b,
                        unsigned int M,
                        unsigned int N,
                        float* p_y) {

    const float* _p_a_iter;
    const float* _p_b_iter;
    float* _p_y_iter = p_y;

    float _acc0, _acc1, _acc2, _acc3;
    float _val_b0, _val_b1, _val_b2, _val_b3;
    float _val_a;

    unsigned int _num_blk = N / 4;
    unsigned int _rem_blk = N % 4;

    for (int _n = 0; _n < _num_blk; _n++) {

        _acc0 = 0.f;
        _acc1 = 0.f;
        _acc2 = 0.f;
        _acc3 = 0.f;
        _p_a_iter = p_a;
        _p_b_iter = p_b + 4 * _n;

        for (int _m = 0; _m < M; _m++) {
            _val_a = *_p_a_iter++;
            _val_b0 = *(_p_b_iter + 0);
            _val_b1 = *(_p_b_iter + 1);
            _val_b2 = *(_p_b_iter + 2);
            _val_b3 = *(_p_b_iter + 3);

            _p_b_iter += N;

            _acc0 = insn_fmadd(_val_a, _val_b0, _acc0);
            _acc1 = insn_fmadd(_val_a, _val_b1, _acc1);
            _acc2 = insn_fmadd(_val_a, _val_b2, _acc2);
            _acc3 = insn_fmadd(_val_a, _val_b3, _acc3);
        }

        *_p_y_iter++ = _acc0;
        *_p_y_iter++ = _acc1;
        *_p_y_iter++ = _acc2;
        *_p_y_iter++ = _acc3;

    }

    for (int _n = 0; _n < _rem_blk; _n++) {

        _acc0 = 0.f;
        _p_a_iter = p_a;
        _p_b_iter = p_b + 4 * _num_blk + _n;

        for (int _n = 0; _n < N; _n++) {
            _val_a = *_p_a_iter++;
            _val_b0 = *(_p_b_iter);

            _p_b_iter += N;

            _acc0 = insn_fmadd(_val_a, _val_b0, _acc0);
        }

        *_p_y_iter++ = _acc0;

    }

}

/**
 * @brief compute vector covariance matrix.
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to vector a of shape [N]
 * @param N Columns of matrix A and B
 * @param store 0: store everything, 1: store upper right half only, 2: store lower left half only
 * @param p_y Pointer to matrix Y = aa^T of shape [N, N], assuming a is a column vector
 */
void linalg_vcovmat_f(const float* p_a,
                      unsigned int N,
                      unsigned int store,
                      float* p_y) {

    const float* _p_a_iter1 = p_a;
    const float* _p_a_iter2;

    float _val_a;

    if (store == 0) {

        /*
         * Store all values of the output matrix
         */

        float* _p_y_iter1;
        float* _p_y_iter2;
        float _val_y;

        for (int _i = 0; _i < N; _i++) {

            // prepare the iterators
            _p_y_iter1 = p_y + _i * (N + 1);
            _p_y_iter2 = _p_y_iter1 + N;
            _p_a_iter2 = _p_a_iter1 + 1;

            // load the first value, which is constant for this entire cycle
            _val_a = *_p_a_iter1++;

            // compute the diagonal element and store back
            *_p_y_iter1++ = _val_a * _val_a;

            // compute the remaining elements
            for (int _j = _i + 1; _j < N; _j++) {
                _val_y = _val_a * (*_p_a_iter2++);
                *_p_y_iter1 = _val_y;
                *_p_y_iter2 = _val_y;
                _p_y_iter1 += 1;
                _p_y_iter2 += N;
            }
        }

    } else if (store == 1) {

        /*
         * Store only the upper half of the values
         */

        float* _p_y_iter;

        for (int _i = 0; _i < N; _i++) {

            // prepare the iterators
            _p_y_iter = p_y + _i * (N + 1);
            _p_a_iter2 = _p_a_iter1 + 1;

            // load the first value, which is constant for this entire cycle
            _val_a = *_p_a_iter1++;

            // compute the diagonal element and store back
            *_p_y_iter++ = _val_a * _val_a;

            // compute the remaining elements
            for (int _j = _i + 1; _j < N; _j++) {
                *_p_y_iter++ = _val_a * (*_p_a_iter2++);
            }
        }

    } else if (store == 2) {

        /*
         * Store only the upper half of the values
         */

        float* _p_y_iter;

        for (int _i = 0; _i < N; _i++) {

            // prepare the iterators
            _p_y_iter = p_y + _i * (N + 1);
            _p_a_iter2 = _p_a_iter1 + 1;

            // load the first value, which is constant for this entire cycle
            _val_a = *_p_a_iter1++;

            // compute the diagonal element and store back
            *_p_y_iter = _val_a * _val_a;
            _p_y_iter += N;

            // compute the remaining elements
            for (int _j = _i + 1; _j < N; _j++) {
                *_p_y_iter = _val_a * (*_p_a_iter2++);
                _p_y_iter += N;
            }
        }

    } else {

        printf("Error in linalg::matop_f::covmat_f, store parameter must be 0, 1 or 2!\n");

    }

}

/**
 * @brief computes the outer product of two vectors
 *
 * @param p_a Pointer to vector a of shape N, is treated as a column vector
 * @param p_b Pointer to vector b of shape M, is treated as a row vector
 * @param N length of vector a and number of rows for matrix Y
 * @param M length of vector b and number of columns of matrix Y
 * @param p_y pointer to matrix Y of shape (N, M)
 */
void linalg_vec_outerprod_f(const float* p_a,
                            const float* p_b,
                            unsigned int N,
                            unsigned int M,
                            float* p_y) {

    const float* _p_a_iter = p_a;
    const float* _p_b_iter;
    float* _p_y_iter = p_y;
    float _val_a;

    unsigned int _num_blk = M / 4;
    unsigned int _rem_blk = M % 4;

    for (int _n = 0; _n < N; _n++) {

        _p_b_iter = p_b;
        _val_a = *p_a++;

        for (unsigned int _m = 0; _m < _num_blk; _m++) {

            *_p_y_iter++ = _val_a * (*(_p_b_iter + 0));
            *_p_y_iter++ = _val_a * (*(_p_b_iter + 1));
            *_p_y_iter++ = _val_a * (*(_p_b_iter + 2));
            *_p_y_iter++ = _val_a * (*(_p_b_iter + 3));
            _p_b_iter += 4;

        }

        for (unsigned int _m = 0; _m < _rem_blk; _m++) {
            *_p_y_iter++ = _val_a * (*_p_b_iter++);
        }

    }

}

/**
 * @brief computes the L2 norm of a given vector a
 *
 * @param p_a pointer to vector a
 * @param N number of elements in vector a
 * @param stride Increment from one element to another, use 1 per default
 * @return l2 norm (sum of the squares)
 */
float linalg_vnorm_f(const float* p_a,
                     unsigned int N,
                     unsigned int stride) {

    /*
     * Loop Unrolling does not improve things because we cannot do load post increment!
     */

    float _acc = 0;
    float _val;

    for (int _i = 0; _i < N; _i++) {

        _val = *p_a;
        p_a += stride;
        _acc = insn_fmadd(_val, _val, _acc);

    }

    _acc = insn_fsqrt(_acc);

    return _acc;

}

/**
 * @brief Computes the transpose of a symmetric matrix A in place
 *
 * @param p_a Pointer to matrix A of shape[N, N]
 * @param N Number of rows and columns of matrix A
 */ 
void linalg_transpose_sf(float* p_a,
                         unsigned int N) {

    float _val_a, _val_b;

    for (int _n = 1; _n < N; _n++) {
        for (int _m = 0; _m < _n; _m++) {
            _val_a = p_a[_n * N + _m];
            _val_b = p_a[_m * N + _n];
            p_a[_n * N + _m] = _val_b;
            p_a[_m * N + _n] = _val_a;
        }
    }
}

/**
 * @brief prints out the matrix to standard out
 *
 * @param p_a Pointer to matrix A
 * @param N number of rows of matrix A
 * @param M number of columns of matrix A
 * @param stride Number of 4-bytes between the start of each row of matrix A, stride >= M
 */
void linalg_print_mat_f(const float* p_a,
                        unsigned int N,
                        unsigned int M,
                        unsigned int stride) {

    const float* _mat_iter = p_a;

    unsigned int line_offset = stride - M;

    printf("[");
    for (int _i = 0; _i < N; _i++) {
        if (_i > 0) {
            printf(",\n ");
        }
        printf("[");

        for (int _j = 0; _j < M; _j++) {
            if (_j > 0) {
                printf(", ");
            }
            printf("%+.2e", *_mat_iter++);
        }

        printf("]");
        _mat_iter += line_offset;
    }
    printf("]\n");

}

/**
 * @brief computes 2 (A + A^T) of the square matrix A with zeros in all rows up to but excluding k
 *
 *    | 0 0 0 0 0 | 
 * A: | x x x x x | k is the index of the first row containing values.
 *    | x x x x x |
 *
 * The first k rows of A are ignored, their value is assumed to be 0.
 *
 * @warning This operation is done inplace. Also, only the upper right nonzero part is updated!
 *
 * @param p_a Pointer to matrix A of shape [N, N], where all rows up to k are ignored
 * @param N Dimensionality of the matrix A
 * @param k Number of rows of A assumed to be 0
 */
void linalg_2aat_f(float* p_a,
                   unsigned int N,
                   unsigned int k) {

    float _val0, _val1;

    // We have 3 different regions, and since the result is symmetric, everything is cloned

    /*
     * Region 1: upper right part, where we only have to multiply all values by 2
     */

    for (int _i = 0; _i < k; _i++) {
        for (int _j = k; _j < N; _j++) {
            _val0 = p_a[_i * N + _j];
            _val0 = 2.f * _val0;
            p_a[_i * N + _j] = _val0;
        }
    }

    /*
     * Region 2 diagonal part
     */
    for (int _i = k; _i < N; _i++) {
        _val0 = p_a[_i * (N + 1)];
        _val0 = 4.f * _val0;
        p_a[_i * (N + 1)] = _val0;
    }

    /*
     * Region 3, lower right part, where we need to load the values, add and store them back
     */

    for (int _i = k; _i < N; _i++) {
        for (int _j = _i + 1; _j < N; _j++) {
            // read the value
            _val0 = p_a[_i * N + _j];
            _val1 = p_a[_j * N + _i];

            // compute the new value
            _val0 = 2.f * (_val0 + _val1);

            // store the value in both places
            p_a[_i * N + _j] = _val0;
        }
    }

}
