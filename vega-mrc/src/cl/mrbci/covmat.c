/**
 * @file covmat.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the Covariance Matrix Block
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
 * @brief Apply the FIR filter for a given frequency
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to filtered input data of shape [C, T], aligned to [C, T_ALIGN]
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C], aligned to [C, C] (assuming C is even)
 */
void mrbci_covmat(const int8_t* p_in,
                  unsigned int freq_idx,
                  int16_t* p_out) {

    int32_t rho = mrbci_covmat_rho[freq_idx];
    unsigned int y_shift = mrbci_covmat_shift[freq_idx];

    func_covmat_reg(p_in, rho, MRBCI_T_ALIGN, MRBCI_C, MRBCI_C, y_shift, p_out);

}
