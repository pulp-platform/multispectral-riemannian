/**
 * @file copy_mat.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation of copy functions
 */

#include "functional.h"
#include "rt/rt_api.h"

/**
 * @brief Copy matrices A to B
 *
 * @param p_src Pointer to matrix A of shape [N, M]
 * @param p_dst Pointer to matrix B of shape [N, M]
 * @param N number of rows of the two matrices
 * @param M number of columns of the two matrices
 * @param stride_src Number of 4-bytes between the start of each row of matrix A, stride_a >= N
 * @param stride_src Number of 4-bytes between the start of each row of matrix B, stride_b >= N
 */
void func_copy_mat(uint32_t* p_src,
                   uint32_t* p_dst,
                   unsigned int N,
                   unsigned int M,
                   unsigned int stride_src,
                   unsigned int stride_dst) {

    uint32_t* _p_src_iter = p_src;
    uint32_t* _p_dst_iter = p_dst;

    uint32_t _val_a, _val_b;

    unsigned int _num_block = M / 2;
    unsigned int _rem_block = M % 2;

    unsigned int _newline_src = stride_src - M;
    unsigned int _newline_dst = stride_dst - M;

    for (int _n = 0; _n < N; _n++) {

        for (int _m = 0; _m < _num_block; _m++) {
            _val_a = *_p_src_iter++;
            _val_b = *_p_src_iter++;
            *_p_dst_iter++ = _val_a;
            *_p_dst_iter++ = _val_b;
        }

        if (_rem_block != 0) {
            *_p_dst_iter++ = *_p_src_iter++;
        }

        _p_src_iter += _newline_src;
        _p_dst_iter += _newline_dst;
    }

}
