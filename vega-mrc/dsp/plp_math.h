#ifndef __PLP_MATH_H__
#define __PLP_MATH_H__

#include "math.h"
#include "rtos_hal.h"

typedef float float32_t;


/** -------------------------------------------------------
 * @brief Instance structure for integer parallel matrix multiplication.
 */
typedef struct {
    const int16_t *__restrict__ pSrcA;
    const int16_t *__restrict__ pSrcB;
    uint32_t M;
    uint32_t N;
    uint32_t O;
    uint32_t nPE;
    int32_t *__restrict__ pDstC;
} plp_mat_mult_instance_i16;


/** -------------------------------------------------------
 * @brief Instance structure for integer parallel matrix multiplication.
 */
typedef struct {
    const int32_t *__restrict__ pSrcA;
    const int32_t *__restrict__ pSrcB;
    uint32_t M;
    uint32_t N;
    uint32_t O;
    uint32_t nPE;
    int32_t *__restrict__ pDstC;
} plp_mat_mult_instance_i32;


/** -------------------------------------------------------
   @brief      Matrix matrix multiplication of a 32-bit integer matrices for RV32IM extension.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[out] pDstC Output is written here
   @return     none
*/

void plp_mat_mult_i32s_rv32im(const int32_t *__restrict__ pSrcA,
                              const int32_t *__restrict__ pSrcB,
                              uint32_t M,
                              uint32_t N,
                              uint32_t O,
                              int32_t *__restrict__ pDstC);

/** -------------------------------------------------------
   @brief      Matrix matrix multiplication of a 32-bit integer matrices for XPULPV2 extension.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[out] pDstC Output is written here
   @return     none
*/

void plp_mat_mult_i32s_xpulpv2(const int32_t *__restrict__ pSrcA,
                               const int32_t *__restrict__ pSrcB,
                               uint32_t M,
                               uint32_t N,
                               uint32_t O,
                               int32_t *__restrict__ pDstC);


/** -------------------------------------------------------
   @brief      Glue code for parallel matrix matrix multiplication of a 32-bit integer matrices.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[in]  nPE   Number of cores to use
   @param[out] pDstC Output is written here
   @return     none
*/

void plp_mat_mult_i32_parallel(const int32_t *__restrict__ pSrcA,
                               const int32_t *__restrict__ pSrcB,
                               uint32_t M,
                               uint32_t N,
                               uint32_t O,
                               uint32_t nPE,
                               int32_t *__restrict__ pDstC);

/** -------------------------------------------------------
   @brief      Parallel matrix matrix multiplication of a 32-bit integer matrices for XPULPV2
               extension.
   @param[in]  args  pointer to plp_mat_mult_instance_i32 struct initialized by
                     plp_mat_mult_i32_parallel
   @return     none
*/

void plp_mat_mult_i32p_xpulpv2(void *args);

/** -------------------------------------------------------
   @brief      Glue code for parallel matrix matrix multiplication of a 16-bit integer matrices.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[in]  nPE   Number of cores to use
   @param[out] pDstC Output is written here
   @return     none
*/

void plp_mat_mult_i16_parallel(const int16_t *__restrict__ pSrcA,
                               const int16_t *__restrict__ pSrcB,
                               uint32_t M,
                               uint32_t N,
                               uint32_t O,
                               uint32_t nPE,
                               int32_t *__restrict__ pDstC);



/** -------------------------------------------------------
    @brief Parallel matrix multiplication of 16-bit integer matrices kernel for XPULPV2 extension.
    @param[in]  args  pointer to plp_mat_mult_instance_i16 struct initialized by
                      plp_mat_mult_i16_parallel
    @return     none

    @par Exploiting SIMD instructions
    The 16 bit values are packed two each into 32 bit vectors and then the two dot products are
    performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_mult_i16p_xpulpv2(void *args);


/** -------------------------------------------------------
   @brief      Glue code for matrix matrix multiplication of a 16-bit integer matrices.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[out] pDstC Output is written here
   @return     none
*/

void plp_mat_mult_i16(const int16_t *__restrict__ pSrcA,
                      const int16_t *__restrict__ pSrcB,
                      uint32_t M,
                      uint32_t N,
                      uint32_t O,
                      int32_t *__restrict__ pDstC);

/** -------------------------------------------------------
   @brief      Matrix matrix multiplication of a 16-bit integer matrices for RV32IM extension.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[out] pDstC Output is written here
   @return     none
*/

void plp_mat_mult_i16s_rv32im(const int16_t *__restrict__ pSrcA,
                              const int16_t *__restrict__ pSrcB,
                              uint32_t M,
                              uint32_t N,
                              uint32_t O,
                              int32_t *__restrict__ pDstC);

/** -------------------------------------------------------
   @brief      Matrix matrix multiplication of a 16-bit integer matrices for XPULPV2 extension.
   @param[in]  pSrcA points to first the input matrix
   @param[in]  pSrcB points to second the input matrix
   @param[in]  M     Height of first matrix
   @param[in]  N     Width of first and heigt of second matrix
   @param[in]  O     Width of second matrix
   @param[out] pDstC Output is written here
   @return     none

   @par Exploiting SIMD instructions
   The 16 bit values are packed two each into 32 bit vectors and then the two dot products are
   performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_mult_i16s_xpulpv2(const int16_t *__restrict__ pSrcA,
                               const int16_t *__restrict__ pSrcB,
                               uint32_t M,
                               uint32_t N,
                               uint32_t O,
                               int32_t *__restrict__ pDstC);




#endif // __PLP_MATH_H__
