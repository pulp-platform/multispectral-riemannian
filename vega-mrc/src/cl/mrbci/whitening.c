/**
 * @file whitening.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the Whitening Block
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

/**
 * @brief Apply the whitening transform (two matrix multiplications)
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C], aligned to [C, C] (assuming C is even)
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C]
 * @param p_workspace Pointer to temporary workspace data of shape [C, C]
 */
void mrbci_whitening(const int16_t* p_in,
                     unsigned int freq_idx,
                     int32_t* p_out,
                     int32_t* p_workspace) {

    const int16_t* c_ref_invsqrtm_i16 = mrbci_c_ref_invsqrtm_i16 + (MRBCI_C * MRBCI_C) * freq_idx;
    const int32_t* c_ref_invsqrtm_i32 = mrbci_c_ref_invsqrtm_i32 + (MRBCI_C * MRBCI_C) * freq_idx;

    func_matmul_sqr_i16(p_in, c_ref_invsqrtm_i16, MRBCI_C, p_workspace);
    func_matmul_sqr_i32(c_ref_invsqrtm_i32, p_workspace, MRBCI_C, p_out);

}
