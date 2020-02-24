/**
 * @file linalg.h
 * @author Tibor Schneider
 * @date 2020/02/21
 * @brief This file contains the definitions for linalg functions
 */

#ifndef __CL_LINALG_H__
#define __CL_LINALG_H__

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
    float cs;
    float sn;
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
 * @param p_y Pointer to matrix Y = AB of shape [M, O]
 */
void linalg_matmul_f(const float* p_a,
                     const float* p_b,
                     unsigned int M,
                     unsigned int N,
                     unsigned int O,
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

#endif //__CL_LINALG_H__
