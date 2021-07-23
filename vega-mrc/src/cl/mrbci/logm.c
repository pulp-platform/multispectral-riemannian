/**
 * @file logm.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the Matrix Logarithm Block
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

#include "mrbci.h"
#include "../func/functional.h"
#include "../linalg/linalg.h"

/**
 * @brief Apply the Matrix Logarithm
 *
 * @warning p_in, p_workspace and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C], is destroyed in this function
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C], aligned to [C, C_ALIGN]
 * @param p_workspace Pointer to temporary workspace data of shape [C, 3*C + 1]
 */
void mrbci_logm(int32_t* p_in,
                unsigned int freq_idx,
                int8_t* p_out,
                float* p_workspace) {

    float dequant_factor = mrbci_logm_dequant[freq_idx];
    float requant_factor = mrbci_logm_requant;

    float* p_float = (float*)p_in;

    // apply dequantization in place
    func_convert_i32_to_f(p_in, p_float, MRBCI_C * MRBCI_C, dequant_factor);

    // compute matrix logarithm
#ifdef PARALLEL
//linalg_logm_parallel(p_float, MRBCI_C, p_workspace);
linalg_logm(p_float, MRBCI_C, p_workspace);
#else //PARALLEL
linalg_logm(p_float, MRBCI_C, p_workspace);
#endif //PARALLEL

// requantize the values
func_convert_f_to_i8(p_float, p_out, MRBCI_C, MRBCI_C, MRBCI_C_ALIGN, requant_factor);

}
