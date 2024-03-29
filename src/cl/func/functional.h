/**
 * @file functional.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the definitions for all main mathematical functions
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

#ifndef __CL_FUNC_FUNCTIONAL_H__
#define __CL_FUNC_FUNCTIONAL_H__

#include "rt/rt_api.h"

/**
 * @struct sos_filt_2S_params
 * @brief Structure containing all the sos filter parameters for a filter with 2 sections
 *
 * @var sos_filt_2S_params::a01 denominator coefficients (a1 and a2) of S0
 * @var sos_filt_2S_params::a11 denominator coefficients (a1 and a2) of S1
 * @var sos_filt_S2_params::b00 numerator coefficient b0 of S0
 * @var sos_filt_S2_params::b01 numerator coefficients (b1 and b2) of S0
 * @var sos_filt_S2_params::b10 numerator coefficient b0 of S1
 * @var sos_filt_S2_params::b11 numerator coefficients (b1 and b2) of S1
 * @var sos_filt_S2_params::shift_a0 Amount to shift the denominators of S0
 * @var sos_filt_S2_params::shift_a1 Amount to shift the denominators of S1
 * @var sos_filt_S2_params::shift_b0 Amount to shift the numerators of S0
 * @var sos_filt_S2_params::shift_b1 Amount to shift the numerators of S1
 * @var sos_filt_S2_params::y_shift Amount to shift the output for renormalization
 */
typedef struct
{
    v2s a01;
    v2s a11;
    int32_t b00;
    v2s b01;
    int32_t b10;
    v2s b11;
    int32_t shift_a0;
    int32_t shift_a1;
    int32_t shift_b0;
    int32_t shift_b1;
    int32_t y_shift;
} func_sos_filt_2S_params_t;

/**
 * @brief Applies a SOS IIR filter to the input data with exactly 2 sections
 *
 * @warning Data must be already present in L1 memory, and the output vector and register vector
 * must be allocated on L1 memory.
 *
 * @param p_x Pointer to input vector on L1 memory
 * @param len Length of input and output vector
 * @param params sos_filt_2S_param structure containing all filter parameters
 * @param p_y Pointer to the output vector.
 */
void func_sos_filt_2S(const int8_t* p_x,
                      unsigned int len,
                      const func_sos_filt_2S_params_t* params,
                      int8_t* p_y);

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
                   unsigned int stride_b);

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
                   unsigned int stride_dst);

/**
 * @brief Copy matrices A to B and transposes it
 *
 * @param p_src Pointer to matrix A of shape [N, N]
 * @param p_dst Pointer to matrix B of shape [N, N]
 * @param N number of rows of the two matrices
 */
void func_copy_transpose_mat(uint32_t* p_src,
                             uint32_t* p_dst,
                             unsigned int N);

/**
 * @brief Convert int32_t to float using a multiplicative factor
 *
 * @param p_src Pointer to source data in int32_t, shape=[N,]
 * @param p_dst Pointer to destination data in float, shape=[N,]
 * @param N Number of elements to convert
 * @param mul_factor Factor to multiply the floating point number
 */
void func_convert_i32_to_f(const int32_t* p_src,
                           float* p_dst,
                           unsigned int N,
                           float mul_factor);

/**
 * @brief Convert float to int8_t using a multiplicative factor
 *
 * @param p_src Pointer to source data in float, shape=[N, M]
 * @param p_dst Pointer to destination data in int8_t, shape=[N,M], will be aligned to [N, stride]
 * @param N Number of rows of the matrix
 * @param M Number of columns of the matrix
 * @param stride Number of elements in each row, this number must be 4-aligned
 * @param mul_factor Factor to multiply the floating point number before converting to int8_t
 */
void func_convert_f_to_i8(const float* p_src,
                          int8_t* p_dst,
                          unsigned int N,
                          unsigned int M,
                          unsigned int stride,
                          float mul_factor);

/**
 * @brief compute the regularized covariance matrix of a matrix X.
 *
 *     Y = X @ X.T
 *
 * @warning p_y must already be allocated on L1
 *
 * @param p_x Pointer to matrix X of shape [N, M]
 * @param rho Regularization parameter, added to the main diagonal elements
 * @param M Number of columns of matrix X
 * @param N Number of rows of matrix X and dimensionality of matrix Y
 * @param N_align Number of 2-aligned columns of matrix Y
 * @param y_shift Number of bits to shift to the right to store Y
 * @param p_y Pointer to output matrix Y of shape [N, N_align]
 */
void func_covmat_reg(const int8_t* p_x,
                     int32_t rho,
                     unsigned int M,
                     unsigned int N,
                     unsigned int N_align,
                     unsigned int y_shift,
                     int16_t* p_y);

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
                         int32_t* p_y);

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
                         int32_t* p_y);


#endif //__CL_FUNC_FUNCTIONAL_H__
