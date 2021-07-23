/**
 * @file convert.c
 * @author Tibor Schneider
 * @date 2020/02/27
 * @brief This file contains the implementation of the convert functions
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
                           float mul_factor) {

    const int32_t* _p_src_iter = p_src;
    float* _p_dst_iter = p_dst;

    int32_t _i0, _i1, _i2, _i3;
    float _f0, _f1, _f2, _f3;

    unsigned int _num_blk = N / 4;
    unsigned int _rem_blk = N % 4;

    for (unsigned int _k = 0; _k < _num_blk; _k++) {
        _i0 = *_p_src_iter++;
        _i1 = *_p_src_iter++;
        _i2 = *_p_src_iter++;
        _i3 = *_p_src_iter++;

        // Hard Cast
        _f0 = (float)_i0;
        _f1 = (float)_i1;
        _f2 = (float)_i2;
        _f3 = (float)_i3;
        
        /*
        // Soft Cast
        _f0 = __floatsisf(_i0);
        _f1 = __floatsisf(_i1);
        _f2 = __floatsisf(_i2);
        _f3 = __floatsisf(_i3);
        */

        _f0 = _f0 * mul_factor;
        _f1 = _f1 * mul_factor;
        _f2 = _f2 * mul_factor;
        _f3 = _f3 * mul_factor;

        *_p_dst_iter++ = _f0;
        *_p_dst_iter++ = _f1;
        *_p_dst_iter++ = _f2;
        *_p_dst_iter++ = _f3;
    }

    for (unsigned int _k = 0; _k < _rem_blk; _k++) {
        _i0 = *_p_src_iter++;

        _f0 = (float)_i0;

        _f0 = _f0 * mul_factor;

        *_p_dst_iter++ = _f0;
    }

}

/**
 * @brief Convert float to int8_t using a multiplicative factor
 *
 * @param p_src Pointer to source data in float, shape=[N, M]
 * @param p_dst Pointer to destination data in int8_t, shape=[N,M], must be aligned to [N, stride]
 * @param N Number of rows of the matrix
 * @param M Number of columns of the matrix
 * @param stride Number of elements in each row, this number must be 4-aligned
 * @param mul_factor Factor to multiply the floating point number before converting to int8_t
 *
 * @todo add rounding
 */
void func_convert_f_to_i8(const float* p_src,
                          int8_t* p_dst,
                          unsigned int N,
                          unsigned int M,
                          unsigned int stride,
                          float mul_factor) {

    const float* _p_src_iter = p_src;
    int8_t* _p_dst_iter = p_dst;

    float _f0, _f1, _f2, _f3;
    int32_t _i0, _i1, _i2, _i3;

    int32_t _blk;

    unsigned int _num_blk = M / 4;
    unsigned int _rem_blk = M % 4;

    unsigned int _new_line = stride - M;

    for (unsigned int _n = 0; _n < N; _n++) {

        for (unsigned int _m = 0; _m < _num_blk; _m++) {
            _f0 = *_p_src_iter++;
            _f1 = *_p_src_iter++;
            _f2 = *_p_src_iter++;
            _f3 = *_p_src_iter++;

            _f0 = _f0 * mul_factor;
            _f1 = _f1 * mul_factor;
            _f2 = _f2 * mul_factor;
            _f3 = _f3 * mul_factor;

            _f0 = _f0 > 0 ? _f0 + 0.5f : _f0 - 0.5f;
            _f1 = _f1 > 0 ? _f1 + 0.5f : _f1 - 0.5f;
            _f2 = _f2 > 0 ? _f2 + 0.5f : _f2 - 0.5f;
            _f3 = _f3 > 0 ? _f3 + 0.5f : _f3 - 0.5f;

            _i0 = (int32_t)_f0;
            _i1 = (int32_t)_f1;
            _i2 = (int32_t)_f2;
            _i3 = (int32_t)_f3;

            _i0 = __CLIP(_i0, 7);
            _i1 = __CLIP(_i1, 7);
            _i2 = __CLIP(_i2, 7);
            _i3 = __CLIP(_i3, 7);

            _blk = (int32_t)__PACK4(_i0, _i1, _i2, _i3);
            *((int32_t*)_p_dst_iter) = _blk;
            _p_dst_iter += 4;

        }

        for (unsigned int _m = 0; _m < _rem_blk; _m++) {
            _f0 = *_p_src_iter++;
            _f0 = _f0 * mul_factor;
            _f0 = _f0 > 0 ? _f0 + 0.5f : _f0 - 0.5f;
            _i0 = (int32_t)_f0;
            _i0 = __CLIP(_i0, 7);
            *_p_dst_iter++ = (int8_t)_i0;
        }

        _p_dst_iter += _new_line;

    }

}
