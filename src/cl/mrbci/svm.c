/**
 * @file svm.c
 * @author Tibor Schneider
 * @date 2020/03/01
 * @brief This file contains the implementation of the SVM
 *
 * @todo Already take in the p_in from l1
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

#if MRBCI_NUM_CLASS != 4
#error "Number of classes must equal 4"
#endif

#define _SVM_LINE_WIDTH (MRBCI_SVM_NUM_FEATURES_ALIGN / 4)

/**
 * @brief compute SVM (matrix multiplication followed by vector addition)
 *
 * @warning p_in and p_out must reside on L2 memory, not on L1.
 *
 * @param p_in Pointer to feature map of shape [MRBCI_SVM_NUM_FEATURES] in L2 (aligned)
 * @param p_out Pointer to result of shape [MRBCI_NUM_CLASS] in L2
 */
void mrbci_svm(const int8_t* p_in,
               int32_t* p_out) {

    rt_dma_copy_t _copy;

    // allocate data on l1
    v4s* _p_in_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
    v4s* _p_in_l1_iter = _p_in_l1;

    // allocate the output on L1
    int32_t* _p_out_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_NUM_CLASS);

    // allocate weight memory
    v4s* _p_weight_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN * MRBCI_NUM_CLASS);
    v4s* _p_weight_l1_iter = _p_weight_l1;

    // instead of allocating the bias, we already start with the bias loaded to p_out

    // make sure that the input contains zeros at the end
    *((int32_t*) _p_in_l1 + MRBCI_SVM_NUM_FEATURES_ALIGN - 1) = 0;

    // load the data
    rt_dma_memcpy((unsigned int)p_in,
                  (unsigned int)_p_in_l1,
                  sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);

    rt_dma_memcpy((unsigned int)mrbci_svm_weights,
                  (unsigned int)_p_weight_l1,
                  sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN * MRBCI_NUM_CLASS,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    rt_dma_memcpy((unsigned int)mrbci_svm_bias,
                  (unsigned int)_p_out_l1,
                  sizeof(int32_t) * MRBCI_NUM_CLASS,
                  RT_DMA_DIR_EXT2LOC, 1, &_copy);

    rt_dma_wait(&_copy);

    // start the computation
    int32_t _acc0 = _p_out_l1[0];
    int32_t _acc1 = _p_out_l1[1];
    int32_t _acc2 = _p_out_l1[2];
    int32_t _acc3 = _p_out_l1[3];

    v4s _v_in;
    v4s _v_w0;
    v4s _v_w1;
    v4s _v_w2;
    v4s _v_w3;

    for (unsigned int _i = 0; _i < _SVM_LINE_WIDTH; _i++) {

        _v_in = *_p_in_l1_iter++;
        _v_w0 = *(_p_weight_l1_iter + 0 * _SVM_LINE_WIDTH);
        _v_w1 = *(_p_weight_l1_iter + 1 * _SVM_LINE_WIDTH);
        _v_w2 = *(_p_weight_l1_iter + 2 * _SVM_LINE_WIDTH);
        _v_w3 = *(_p_weight_l1_iter + 3 * _SVM_LINE_WIDTH);

        _p_weight_l1_iter++;

        _acc0 = __SUMDOTP4(_v_in, _v_w0, _acc0);
        _acc1 = __SUMDOTP4(_v_in, _v_w1, _acc1);
        _acc2 = __SUMDOTP4(_v_in, _v_w2, _acc2);
        _acc3 = __SUMDOTP4(_v_in, _v_w3, _acc3);

    }

    //write values back into array
    _p_out_l1[0] = _acc0;
    _p_out_l1[1] = _acc1;
    _p_out_l1[2] = _acc2;
    _p_out_l1[3] = _acc3;

    // copy the result back to l2
    rt_dma_memcpy((unsigned int)p_out,
                  (unsigned int)_p_out_l1,
                  sizeof(int32_t) * MRBCI_NUM_CLASS,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free up all the memory
    rt_free(RT_ALLOC_CL_DATA, _p_in_l1, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
    rt_free(RT_ALLOC_CL_DATA, _p_out_l1, sizeof(int32_t) * MRBCI_NUM_CLASS);
    rt_free(RT_ALLOC_CL_DATA, _p_weight_l1, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN * MRBCI_NUM_CLASS);

}
