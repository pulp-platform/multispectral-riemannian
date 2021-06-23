/**
 * @file swap_mat.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation of swap functions
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
 * @brief Swap matrices A and B
 *
 * @param p_a Pointer to matrix A of shape [N, M]
 * @param p_b Pointer to matrix B of shape [N, M]
 * @param N number of rows of the two matrices
 * @param M number of columns of the two matrices
 * @param stride_a Number of 4-bytes between the start of each row of matrix A, stride_a >= N
 * @param stride_b Number of 4-bytes between the start of each row of matrix B, stride_b >= N
 */
void func_swap_mat(uint32_t* p_a,
                   uint32_t* p_b,
                   unsigned int N,
                   unsigned int M,
                   unsigned int stride_a,
                   unsigned int stride_b) {

    uint32_t* _p_a_iter = p_a;
    uint32_t* _p_b_iter = p_b;

    uint32_t _val_a;
    uint32_t _val_b;

    unsigned int newline_a = stride_a - M;
    unsigned int newline_b = stride_b - M;

    for (int _n = 0; _n < N; _n++) {
        for (int _m = 0; _m < M; _m++) {
            _val_a = *_p_a_iter;
            _val_b = *_p_b_iter;
            *_p_a_iter++ = _val_b;
            *_p_b_iter++ = _val_a;
        }
        _p_a_iter += newline_a;
        _p_b_iter += newline_b;
    }

}
