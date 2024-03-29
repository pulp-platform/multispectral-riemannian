/**
 * @file linalg.h
 * @author Tibor Schneider
 * @date 2020/02/21
 * @brief This file contains the definitions for linalg functions
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

#ifndef __CL_LINALG_H__
#define __CL_LINALG_H__

/**
 * @brief Compute the matrix logarithm of a matrix by computing the SVD first.
 *
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the matrix logarithm of A
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (2 * N + 2) space
 */
void linalg_logm(float* p_a,
                 unsigned int N,
                 float* p_workspace);

/**
 * @brief Compute the matrix logarithm of a matrix in parallel by computing the SVD first.
 *
 * @param p_a Pointer to matrix A of shape [N, N], must be symmetric. After returning, this matrix
 *            contains the matrix logarithm of A
 * @param N Dimension of matrix A
 * @param p_workspace Temporary storage required for computation, requires (N * (2 * N + 2) space
 */
void linalg_logm_parallel(float* p_a,
                          unsigned int N,
                          float* p_workspace);

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
 * @param p_workspace Temporary storage required for computation, requires (N * 2) space
 */
void linalg_svd_sym(float* p_a,
                    float* p_q,
                    unsigned int N,
                    float* p_workspace);

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
                            unsigned int current_pos);

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
                                    float* p_workspace);

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
                                    unsigned int kp1);


/**
 * @brief updates matrix Q inside the householder tridiagonalization
 *
 *     Q = Q - 2 * w v^T  (both v and w are assumed to be column vectors)
 *
 * @param p_q Pointer to matrix Q, of shape [N, N], is updated in place
 * @param p_v Pointer to vector v of shape [N], all values up to k+1 are assumed to be zero
 * @param p_w Pointer to vector w of shape [N]
 * @param N Dimensionality of Q and both vectors v and w
 * @param kp1 Part of the matrices which are zero (k + 1)
 */
void linalg_householder_update_step_Q(float* p_q,
                                      const float* p_v,
                                      const float* p_w,
                                      unsigned int N,
                                      unsigned int kp1);

/**
 * @struct linalg_givens_rotation_t
 * @brief structure containing the sine and cosine for a givens rotation
 *
 * @var linalg_givens_rotation_t::cs cosine for the rotation
 * @var linalg_givens_rotation_t::cs sine for the rotation
 */
typedef struct {
    float cs;
    float sn;
} linalg_givens_rotation_t;

/**
 * @brief Computes the givens rotation coefficients cosine and sine
 *
 * | cs -sn | | a | = | r |
 * | sn  cs | | b | = | 0 |
 *
 * @param a first value
 * @param b second value
 * @returns linalg_givens_rotation_t structure
 */
linalg_givens_rotation_t linalg_givens_rotation(float a,
                                                float b);

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
                                                     float c);

/**
 * @struct linalg_evd_2x2_t
 * @brief structure containing both the rotation and the eigenvalues of a 2x2 symmetric matrix.
 *
 * @var linalg_evd_2x2_t::cs cosine for the rotation
 * @var linalg_evd_2x2_t::cs sine for the rotation
 */
typedef struct {
    float ev1;
    float ev2;
    linalg_givens_rotation_t rot;
} linalg_evd_2x2_t;

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
                                float c);

/**
 * @brief fills matrix A with the unit matrix, with one on the main diagonal.
 *
 * @param p_a: pointer to matrix A of shape (dim, dim). Must be allocated
 * @param dim: dimension of matrix A
 */
void linalg_fill_I(float* p_a, unsigned int dim);

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
                     float* p_y);

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
                     float* p_y);

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
                            float* p_y);

/**
 * @brief Compute the matrix multiplication of two floating point matrices in parallel
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param core_id Id of the calling core
 * @param num_workers number of cores working in parallel
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
void linalg_matmul_stride_parallel_f(unsigned int core_id,
                                     unsigned int num_workers,
                                     const float* p_a,
                                     const float* p_b,
                                     unsigned int M,
                                     unsigned int N,
                                     unsigned int O,
                                     unsigned int stride_a,
                                     unsigned int stride_b,
                                     unsigned int stride_y,
                                     float* p_y);

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
inline void linalg_matmul_f(const float* p_a,
                            const float* p_b,
                            unsigned int M,
                            unsigned int N,
                            unsigned int O,
                            float* p_y) {
    linalg_matmul_stride_f(p_a, p_b, M, N, O, N, O, O, p_y);
}

/**
 * @brief Compute the matrix multiplication of two floating point matrices, where the result
 * is already known to be symmetric (used in Householder Transformation)
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to matrix B of shape [N, O]
 * @param N Rows and columns of all matrices
 * @param p_y Pointer to matrix Y = AB of shape [M, O]
 */
void linalg_matmul_to_sym_f(const float* p_a,
                            const float* p_b,
                            unsigned int N,
                            float* p_y);

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
                       float* p_y);

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
                          unsigned int N);

/**
 * @brief computes the matrix multiplication of a matrix A and a diagonal matrix D.
 *
 *     A <-- A @ D
 *
 * @warning The matrix A will be overwritten with the result
 *
 * @param core_id Id of the calling core
 * @param num_workers number of cores working in parallel
 * @param p_a Pointer to matrix A of shape [N, N], will be overwritten
 * @param p_diag Pointer to diagonal vector of matrix D, of shape [N]
 * @param N Dimension of the matrix A and length of diagonal vector D
 */
void linalg_matmul_diag_parallel_f(unsigned int core_id,
                                   unsigned int num_workers,
                                   float* p_a,
                                   const float* p_diag,
                                   unsigned int N);

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
                        float* p_y);

/**
 * @brief Compute the matrix vector multiplication in parallel. b is assumed to be a column vector
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param core_id id of the current core
 * @param num_workers number of cores computing this function
 * @param p_a Pointer to matrix A of shape [M, N]
 * @param p_b Pointer to vector b of shape [N]
 * @param M Rows of matrix A and length of vector y
 * @param N columns of matrix A and length of vector b
 * @param stride_a number of elements between the beginning of each row of matrix A, stride_a >= N
 * @param p_y Pointer to vector y = Ab of shape [M]
 */
void linalg_matvecmul_parallel_f(unsigned int core_id,
                                 unsigned int num_workers,
                                 const float* p_a,
                                 const float* p_b,
                                 unsigned int M,
                                 unsigned int N,
                                 unsigned int stride_a,
                                 float* p_y);

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
                        float* p_y);

/**
 * @brief Compute the vector matrix multiplication in parallel. Vector a is assumed to be a row vector
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param core_id id of the current core
 * @param num_workers number of cores computing this function
 * @param p_a Pointer to vector a of shape [M]
 * @param p_b Pointer to matrix B of shape [M, N]
 * @param M length of vector a and columns of matrix B
 * @param N rows of matrix B and length of vector y
 * @param p_y Pointer to vector y = Ab of shape [N]
 */
void linalg_vecmatmul_parallel_f(unsigned int core_id,
                                 unsigned int num_workers,
                                 const float* p_a,
                                 const float* p_b,
                                 unsigned int M,
                                 unsigned int N,
                                 float* p_y);

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
                                    unsigned int N);

/**
 * @brief Computes the transpose of a symmetric matrix A in place
 *
 * @param p_a Pointer to matrix A of shape[N, N]
 * @param N Number of rows and columns of matrix A
 */ 
void linalg_transpose_sf(float* p_a,
                         unsigned int N);

/**
 * @brief compute vector covariance matrix.
 *
 * @warning p_y must already be allocated, use L1 memory!
 *
 * @param p_a Pointer to vector a of shape [N]
 * @param N Columns of matrix A and B
 * @param stride Line width of the output matrix Y
 * @param store 0: store everything, 1: store upper right half only, 2: store lower left half only
 * @param p_y Pointer to matrix Y = aa^T of shape [N, N], assuming a is a column vector
 */
void linalg_vcovmat_f(const float* p_a,
                      unsigned int N,
                      unsigned int stride,
                      unsigned int store,
                      float* p_y);

/**
 * @brief computes the inner product of two vectors
 *
 * @param p_a Pointer to vector a of shape N, is treated as a row vector
 * @param p_b Pointer to vector b of shape N, is treated as a column vector
 * @param N length of vector a and b
 */
float linalg_vec_innerprod_f(const float* p_a,
                             const float* p_b,
                             unsigned int N);

/**
 * @brief computes the outer product of two vectors
 *
 * @param p_a Pointer to vector a of shape N, is treated as a column vector
 * @param p_b Pointer to vector b of shape M, is treated as a row vector
 * @param N length of vector a and number of rows for matrix Y
 * @param M length of vector b and number of columns of matrix Y
 * @param stride Distance for each line when writing out the matrix Y
 * @param p_y pointer to matrix Y of shape (N, M)
 */
void linalg_vec_outerprod_f(const float* p_a,
                            const float* p_b,
                            unsigned int N,
                            unsigned int M,
                            unsigned int stride,
                            float* p_y);

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
                     unsigned int stride);

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
                     unsigned int stride);

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
                        unsigned int stride);

#endif //__CL_LINALG_H__
