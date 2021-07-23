/**
 * @file half_diag.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for half diagonalization and renormalizing
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

#include "rt/rt_api.h"
#include "mrbci.h"

#if MRBCI_C % 2 != 0
#error "C must be even!"
#endif

/**
 * @brief Convert the upper right half of the matrix into a 
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C_ALIGN], is destroyed in this function
 * @param p_out Pointer to output data, must at least MRBCI_HALF_DIAG_FEATURES space
 *
 * @todo currently, we are using a very slow algorithm without any loop unrolling. This should be changed
 */
void mrbci_half_diag(const int8_t* p_in,
                     int8_t* p_out) {

    int8_t* _p_out_iter = p_out;

    int32_t _val_0, _val_1;

    // do the main diagonal
    for (int _i = 0; _i < MRBCI_C / 2; _i++) {
        int _j1 = _i * 2;
        int _j2 = _i * 2 + 1;
        _val_0 = p_in[_j1 * MRBCI_C_ALIGN + _j1];
        _val_1 = p_in[_j2 * MRBCI_C_ALIGN + _j2];

#if MRBCI_HALF_DIAG_SHIFT_DIAG > 0
        _val_0 = __ROUNDNORM(_val_0, MRBCI_HALF_DIAG_SHIFT_DIAG);
        _val_1 = __ROUNDNORM(_val_1, MRBCI_HALF_DIAG_SHIFT_DIAG);
#endif

        *p_out++ = (int8_t)_val_0;
        *p_out++ = (int8_t)_val_1;
    }

    // do all other values
    for (int _i = 1; _i < MRBCI_C; _i++) {

        int _num_blk = _i / 2;
        int _rem_blk = _i % 2;

        for (int _j = 0; _j < _num_blk; _j++) {

            int _row_0 = _j * 2;
            int _row_1 = _j * 2 + 1;

            _val_0 = p_in[_i * MRBCI_C_ALIGN + _row_0];
            _val_1 = p_in[_i * MRBCI_C_ALIGN + _row_1];

            _val_0 = __ROUNDNORM(_val_0 * MRBCI_HALF_DIAG_SQRT2, MRBCI_HALF_DIAG_SHIFT);
            _val_1 = __ROUNDNORM(_val_1 * MRBCI_HALF_DIAG_SQRT2, MRBCI_HALF_DIAG_SHIFT);

            *p_out++ = (int8_t)_val_0;
            *p_out++ = (int8_t)_val_1;

        }

        // do the remaining
        if (_rem_blk > 0) {

            _val_0 = p_in[_i * MRBCI_C_ALIGN + _i - 1];
            _val_0 = __ROUNDNORM(_val_0 * MRBCI_HALF_DIAG_SQRT2, MRBCI_HALF_DIAG_SHIFT);
            *p_out++ = (int8_t)_val_0;

        }
    }

}
