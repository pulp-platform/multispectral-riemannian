/**
 * @file mrbci.c
 * @author Tibor Schneider
 * @date 2020/02/28
 * @brief This file contains the init and the main run function for the MRBCI
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
#include "mrbci_params.h"
#include "../func/functional.h"

/**
 * @brief initializes mrbci
 */
void mrbci_init() {
    mrbci_logm_dequant = (float*)mrbci_logm_dequant_i;
    mrbci_logm_requant = *(float*)(&mrbci_logm_requant_i);
}


/**
 * @brief Do the entire multiscale reiamnnian BCI
 *
 * @brief p_in and p_out are not allowed to be in L1, they must reside in L2 memory.
 *
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param p_out Pointer to result of shape [MRBCI_NUM_CLASS] in L2
 */
void mrbci_run(const int8_t* p_in,
               int32_t* p_out) {

    // Allocate the output vector on L2
    int8_t* _feature_map = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

    // compute all the features
    mrbci_extract_features(p_in, _feature_map);

    // compute the output
    mrbci_svm(_feature_map, p_out);

    // free up the allocated memory
    rt_free(RT_ALLOC_FC_DATA, _feature_map, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
}
