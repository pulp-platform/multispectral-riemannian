/**
 * @file functional.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the definitions for all main mathematical functions
 */

#ifndef __CL_FUNC_FUNCTIONAL_H__
#define __CL_FUNC_FUNCTIONAL_H__

#include "rt/rt_api.h"

/**
 * @brief Applies a SOS IIR filter to the input data
 *
 * @warning Data must be already present in L1 memory, and the output vector and register vector
 * must be allocated on L1 memory.
 *
 * @warning For each section of the numerator and denominator, the three coefficients must be stored
 * in the following way:
 *     a[0] = a0, a[1] = X, a[2] = a1, a[3] = a2
 *     b[0] = b0, b[1] = X, b[2] = a1, b[3] = a2
 * The reason for this is that we can do a simd multiply and accumulate with the registers
 *
 * @param p_x Pointer to input vector on L1 memory
 * @param len Length of input and output vector
 * @param p_a Pointer to denominator coefficients, of length (n_section * 4)
 * @param p_b Pointer to numerator coefficients, of length (n_sections * 4)
 * @param p_a_shift Pointer to bitshift vector for denominator, of length n_sections
 * @param p_a_shift Pointer to bitshift vector for numerator, of length n_sections
 * @param n_sections Number of sections
 * @param y_shift Number of bits to shift the output
 * @param regs Pointer to register memory, of length (n_sections * 2), will be modified
 * @param p_y Pointer to the output vector.
 */
void func_sos_filt(const int8_t* p_x,
                   unsigned int len,
                   const int16_t* p_a,
                   const int16_t* p_b,
                   const int8_t* p_a_shift,
                   const int8_t* p_b_shift,
                   unsigned int n_sections,
                   int8_t y_shift,
                   int16_t* regs,
                   int8_t* p_y);


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



#endif //__CL_FUNC_FUNCTIONAL_H__
