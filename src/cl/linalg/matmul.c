/**
 * @file matmul.c
 * @author Tibor Schneider
 * @date 2020/02/22
 * @brief This file contains the functions for floating point matrix multiplication
 */

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

    // use iterators
    const float* _p_a_iter;
    const float* _p_b_iter;
    float* _p_y_iter = p_y;

    float _acc;

    // loop over every output element
    for (unsigned int _m = 0; _m < M; _m++) {
        for (unsigned int _o = 0; _o < O; _o++) {

            _p_a_iter = p_a + _m * N;
            _p_b_iter = p_b + _o;
            _acc = 0;

            for (unsigned int _n = 0; _n < N; _n++) {
                _acc += (*_p_a_iter) * (*_p_b_iter);
                _p_a_iter++;
                _p_b_iter += O;
            }

            *(_p_y_iter++) = _acc;
        }
    }

}
