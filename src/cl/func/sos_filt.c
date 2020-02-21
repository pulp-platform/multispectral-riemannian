/**
 * @file sos_filt.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the SOS IIR filter functions
 */

#include "functional.h"
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
 * @param regs Pointer to register memory, of length ((n_sections + 1) * 2), will be modified
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
                   int8_t* p_y) {

    // TODO implement

}

#define _ZERO2 = (v2s){0, 0};

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
                      int8_t* p_y) {

    // define iterators
    const int8_t* _p_x_iter = p_x;
    int8_t* _p_y_iter = p_y;

    // create local registers
    v2s _reg0 = {0, 0};
    v2s _reg1 = {0, 0};
    v2s _reg2 = {0, 0};
    int32_t _x0;
    int32_t _x1;
    int32_t _x2;

    int32_t _tmp;

    // create shifts
    int _shift_a0 = params->shift_a0;
    int _shift_a1 = params->shift_a1;
    int _shift_b0 = params->shift_b0;
    int _shift_b1 = params->shift_b1;

    // create coefficients
    v2s _a01 = params->a01;
    v2s _a11 = params->a11;

    int32_t _b00 = params->b00;
    v2s _b01 = params->b01;
    int16_t _b10 = params->b10;
    v2s _b11 = params->b11;

    int32_t _y_shift = params->y_shift;

    // start the loop
    for (int _k = 0; _k < len; _k++) {

        // Naive approach of loading every element of x one by one
        _x0 = (int32_t)(*_p_x_iter++);

        // Seciton 1
        _x1 = _x0 * _b00;
        _x1 = __SUMDOTP2(_reg0, _b01, _x1);
        _x1 = _x1 >> _shift_b0;
        _tmp = __DOTP2(_reg1, _a01) >> _shift_a0;
        _x1 -= _tmp;

        // Section 2
        _x2 = _x1 * _b10;
        _x2 = __SUMDOTP2(_reg1, _b11, _x2);
        _x2 = _x2 >> _shift_b1;
        _tmp = __DOTP2(_reg2, _a11) >> _shift_a0;
        _x2 -= _tmp;

        // store the output
        *_p_y_iter++ = _x2 >> _y_shift;

        // suffle around the registers
        _reg0 = __builtin_shuffle(_reg0, (v2s)_x0, (v2s){2, 0});
        _reg1 = __builtin_shuffle(_reg1, (v2s)_x1, (v2s){2, 0});
        _reg2 = __builtin_shuffle(_reg2, (v2s)_x2, (v2s){2, 0});

    }
}
