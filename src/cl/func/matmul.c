/**
 * @file matmul.c
 * @author Tibor Schneider
 * @date 2020/02/27
 * @brief This file contains the implementation of the matmul functions
 *
 * We are using pulp_dsp library currently to do those matmuls.
 * If it turns out that those take up a significant protion of the computation, we should rewrite
 * them, because they can be further optimized.
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
#include "plp_math.h"

#ifndef NUM_WORKERS
#define NUM_WORKERS 9
#endif//NUM_WORKERS

/**
 * @brief compute matrix multiplication of two square int16 matrices
 *
 * @warning N must be even! Also, p_y must already be allocated
 *
 * @param p_a Pointer to matrix A, of shape [N, N]
 * @param p_b Pointer to matrix B, of shape [N, N]
 * @param N Dimensionality of matrices
 * @param p_y Pointer to matrix Y, of shape [N, N]
 */
void func_matmul_sqr_i16(const int16_t* p_a,
                         const int16_t* p_b,
                         unsigned int N,
                         int32_t* p_y) {

#ifdef PARALLEL
    plp_mat_mult_i16_parallel(p_a, p_b, N, N, N, NUM_WORKERS, p_y);
#else //PARALLEL
    plp_mat_mult_i16(p_a, p_b, N, N, N, p_y);
#endif //PARALLEL

}

/**
 * @brief compute matrix multiplication of two square int32 matrices
 *
 * @warning p_y must already be allocated
 *
 * @param p_a Pointer to matrix A, of shape [N, N]
 * @param p_b Pointer to matrix B, of shape [N, N]
 * @param N Dimensionality of matrices
 * @param p_y Pointer to matrix Y, of shape [N, N]
 */
void func_matmul_sqr_i32(const int32_t* p_a,
                         const int32_t* p_b,
                         unsigned int N,
                         int32_t* p_y) {

#ifdef PARALLEL
    plp_mat_mult_i32_parallel(p_a, p_b, N, N, N, NUM_WORKERS, p_y);
#else //PARALLEL
    plp_mat_mult_i32(p_a, p_b, N, N, N, p_y);
#endif //PARALLEL

}
