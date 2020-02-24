/**
 * @file matmul.c
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
 * @param p_y Pointer to matrix Y = AB of shape [M, O]
 */
void linalg_matmul_f(const float* p_a,
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

                _acc0 = insn_fmadd(_val_a, *(_p_b_iter_comp + 0), _acc0);
                _acc1 = insn_fmadd(_val_a, *(_p_b_iter_comp + 1), _acc1);
                _acc2 = insn_fmadd(_val_a, *(_p_b_iter_comp + 2), _acc2);
                _acc3 = insn_fmadd(_val_a, *(_p_b_iter_comp + 3), _acc3);

                _p_a_iter_comp++;
                _p_b_iter_comp += O;
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
                _p_b_iter_comp += O;
            }

            *(_p_y_iter++) = _acc0;
            _p_b_iter++;

        }
        _p_a_iter += N;
        _p_b_iter = p_b;
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

