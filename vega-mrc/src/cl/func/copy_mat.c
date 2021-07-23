/**
 * @file copy_mat.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation of copy functions
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

/**
 * @brief Copy matrices A to B and transposes it
 *
 * @param p_src Pointer to matrix A of shape [N, N]
 * @param p_dst Pointer to matrix B of shape [N, N]
 * @param N number of rows of the two matrices
 */
void func_copy_transpose_mat(uint32_t* p_src,
                             uint32_t* p_dst,
                             unsigned int N) {

    uint32_t* _p_src_iter = p_src;
    uint32_t* _p_dst_iter;

    uint32_t _val_a, _val_b;

    unsigned int _num_block = N / 2;
    unsigned int _rem_block = N % 2;

    for (int _n = 0; _n < N; _n++) {

        _p_dst_iter = p_dst + _n;

        for (int _m = 0; _m < _num_block; _m++) {
            _val_a = *_p_src_iter++;
            _val_b = *_p_src_iter++;
            *(_p_dst_iter + 0 * N) = _val_a;
            *(_p_dst_iter + 1 * N) = _val_b;
            _p_dst_iter += 2 * N;
        }

        if (_rem_block != 0) {
            *_p_dst_iter++ = *_p_src_iter++;
            _p_dst_iter += N;
        }
    }

}
