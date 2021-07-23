/**
 * @file insn.h
 * @author Tibor Schneider
 * @date 2020/02/22
 * @brief This file contains inline functions to call instructions on RISCY
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

#ifndef __INSN_H__
#define __INSN_H__

#include "rt/rt_api.h"
#include "math.h"

/**
 * @defgroup insn_float Floating Point Instructions
 * @{
 */

/**
 * @brief calls fadd.s, 2 cycles latency
 * @param a float
 * @param b float
 * @returns a + b
 */
inline float insn_fadd(float a, float b) {
    float y;
    asm ("fadd.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fsub.s, 2 cycles latency
 * @param a float
 * @param b float
 * @returns a - b
 */
inline float insn_fsub(float a, float b) {
    float y;
    asm ("fsub.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fmul.s, 2 cycles latency
 * @param a float
 * @param b float
 * @returns a * b
 */
inline float insn_fmul(float a, float b) {
    float y;
    asm ("fmul.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fdiv.s, 5-8 cycles latency
 * @param a float
 * @param b float
 * @returns a / b
 */
inline float insn_fdiv(float a, float b) {
    float y;
    asm ("fdiv.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fmadd.s, 3 cycles latency
 * @param a float
 * @param b float
 * @param c float
 * @returns a * b + c
 */
inline float insn_fmadd(float a, float b, float c) {
#ifdef USE_FUSED_FPU
    float y;
    asm ("fmadd.s %[y],%[a],%[b],%[c];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b), [c] "f" (c));
    return y;
#else //USE_FUSED_FPU
    return insn_fadd(insn_fmul(a, b), c);
#endif //USE_FUSED_FPU
}

/**
 * @brief calls fmsub.s, 3 cycles latency
 * @param a float
 * @param b float
 * @param c float
 * @returns a * b - c
 */
inline float insn_fmsub(float a, float b, float c) {
#ifdef USE_FUSED_FPU
    float y;
    asm ("fmsub.s %[y],%[a],%[b],%[c];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b), [c] "f" (c));
    return y;
#else //USE_FUSED_FPU
    return insn_fsub(insn_fmul(a, b), c);
#endif //USE_FUSED_FPU
}

/**
 * @brief calls fnmadd.s, 3 cycles latency
 * @param a float
 * @param b float
 * @param c float
 * @returns -(a * b + c)
 */
inline float insn_fnmadd(float a, float b, float c) {
#ifdef USE_FUSED_FPU
    float y;
    asm ("fnmadd.s %[y],%[a],%[b],%[c];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b), [c] "f" (c));
    return y;
#else //USE_FUSED_FPU
    return -insn_fadd(insn_fmul(a, b), c);
#endif //USE_FUSED_FPU
}

/**
 * @brief calls fnmsub.s, 3 cycles latency
 * @param a float
 * @param b float
 * @param c float
 * @returns -(a * b - c)
 */
inline float insn_fnmsub(float a, float b, float c) {
#ifdef USE_FUSED_FPU
    float y;
    asm ("fnmsub.s %[y],%[a],%[b],%[c];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b), [c] "f" (c));
    return y;
#else //USE_FUSED_FPU
    return -insn_fsub(insn_fmul(a, b), c);
#endif //USE_FUSED_FPU
}

/**
 * @brief calls fsqrt.s, 5-8 cycles latency
 * @param a float
 * @returns sqrt(a)
 */
inline float insn_fsqrt(float a) {
#ifdef USE_SOFT_SQRTDIV
    return sqrtf(a);
#else //USE_SOFT_SQRTDIV
    float y;
    asm ("fsqrt.s %[y],%[a];"
         : [y] "=&f" (y)
         : [a] "f" (a));
    return y;
#endif //USE_SOFT_SQRTDIV
}

/**
 * @brief calls fsqrt.s, 5-8 cycles latency
 * @param a float
 * @returns sqrt(a)
 */
inline float insn_fpow2(float a) {
    float y;
    asm ("fmul.s %[y],%[a],%[a];"
         : [y] "=&f" (y)
         : [a] "f" (a));
    return y;
}

/**
 * @brief calls fabs.s, (probably) 1 cycle latency
 * @param a float
 * @returns |a|
 */
inline float insn_fabs(float a) {
    float y;
    asm ("fabs.s %[y],%[a];"
         : [y] "=&f" (y)
         : [a] "f" (a));
    return y;
}

/**
 * @brief calls fsgnj.s, 1 cycle latency
 * @param a float
 * @param b float
 * @returns |a| * sign(b)
 */
inline float insn_fsgnj(float a, float b) {
    float y;
    asm ("fsgnj.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fsgnjn.s, 1 cycle latency
 * @param a float
 * @param b float
 * @returns -|a| * sign(b)
 */
inline float insn_fsgnjn(float a, float b) {
    float y;
    asm ("fsgnjn.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fsgnjx.s, 1 cycle latency
 * @param a float
 * @param b float
 * @returns a * sign(b)
 */
inline float insn_fsgnjx(float a, float b) {
    float y;
    asm ("fsgnjx.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fmax.s, 1 cycle latency
 * @param a float
 * @param b float
 * @returns max(a, b)
 */
inline float insn_fmax(float a, float b) {
    float y;
    asm ("fmax.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @brief calls fmin.s, 1 cycle latency
 * @param a float
 * @param b float
 * @returns max(a, b)
 */
inline float insn_fmin(float a, float b) {
    float y;
    asm ("fmin.s %[y],%[a],%[b];"
         : [y] "=&f" (y)
         : [a] "f" (a), [b] "f" (b));
    return y;
}

/**
 * @}
 */

#endif//__INSN_H__
