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

#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/mrbci/mrbci.h"
#include "test_stimuli.h"

#ifndef NUM_WORKERS
#define NUM_WORKERS 9
#endif//NUM_WORKERS
#define _LOGM_WORKSPACE_SIZE (MRBCI_C * (2 * MRBCI_C + 2))

// extern definitions, since those are not exported in mrbci.h
typedef struct {
    int32_t* p_whitened;
    float* p_logm_workspace;
    int8_t* p_logm_out;
    int8_t* p_out;
    unsigned int start_freq_idx;
} _mrbci_feature_extraction_kernel_instance_t;
void _mrbci_feature_extraction_kernel(void* args);

RT_CL_DATA static float* workspace_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    int8_t* y_acq = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
    rt_dma_copy_t _copy;

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);



    int32_t* _p_whitened_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C * NUM_WORKERS);
    float* _p_logm_workspace_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * _LOGM_WORKSPACE_SIZE * NUM_WORKERS);
    int8_t* _p_logm_out_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN * NUM_WORKERS);
    int8_t* _p_out_l1_th = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_HALF_DIAG_FEATURES * NUM_WORKERS);

    int _merge = 0;

    for (unsigned int _start_freq_idx = 0; _start_freq_idx < MRBCI_NUM_FREQ; _start_freq_idx += NUM_WORKERS) {

        unsigned int _batch_size = MRBCI_NUM_FREQ - _start_freq_idx;
        if (_batch_size > NUM_WORKERS) {
            _batch_size = NUM_WORKERS;
        }

        // copy the whitened data to l1
        rt_dma_memcpy((unsigned int)(x_stm + _start_freq_idx * MRBCI_C * MRBCI_C),
                      (unsigned int)_p_whitened_l1_th,
                      sizeof(int32_t) * MRBCI_C * MRBCI_C * _batch_size,
                      RT_DMA_DIR_EXT2LOC, _merge, &_copy);

        // prepare the arguments for the kernel
        _mrbci_feature_extraction_kernel_instance_t _args;
        _args.p_whitened = _p_whitened_l1_th;
        _args.p_logm_workspace = _p_logm_workspace_l1_th;
        _args.p_logm_out = _p_logm_out_l1_th;
        _args.p_out = _p_out_l1_th;
        _args.start_freq_idx = _start_freq_idx;

        // wait until the dma is finished
        rt_dma_wait(&_copy);

        // call the cluster
        rt_team_fork(NUM_WORKERS, _mrbci_feature_extraction_kernel, &_args);

        // copy the data back to the result
        rt_dma_memcpy((unsigned int)(y_acq + MRBCI_HALF_DIAG_FEATURES * _start_freq_idx),
                      (unsigned int)_p_out_l1_th,
                      sizeof(int8_t) * MRBCI_HALF_DIAG_FEATURES * _batch_size,
                      RT_DMA_DIR_LOC2EXT, 0, &_copy);

        // remember that the next memcopy to l1 must be merged
        _merge = 1;

    }

    // wait until the data is copied
    rt_dma_wait(&_copy);

    // free the data required in the kernel
    rt_free(RT_ALLOC_CL_DATA, _p_whitened_l1_th, sizeof(int32_t) * MRBCI_C * MRBCI_C * NUM_WORKERS);
    rt_free(RT_ALLOC_CL_DATA, _p_logm_workspace_l1_th, sizeof(float) * _LOGM_WORKSPACE_SIZE * NUM_WORKERS);
    rt_free(RT_ALLOC_CL_DATA, _p_logm_out_l1_th, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN * NUM_WORKERS);
    rt_free(RT_ALLOC_CL_DATA, _p_out_l1_th, sizeof(int8_t) * MRBCI_HALF_DIAG_FEATURES * NUM_WORKERS);



    rt_perf_stop(perf);

    int error = 0;
    for (int i = 0; i < MRBCI_SVM_NUM_FEATURES; i++) {
            if (y_acq[i] != y_exp[i]) {
                error = 1;
                //printf("error at: %d: acq=%d, exp=%d\n", i, y_acq[i], y_exp[i]);
            }
    }

    rt_free(RT_ALLOC_FC_DATA, y_acq, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

    return error;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    mrbci_init();

    int result;

    result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## 1: result: OK\n");
    } else {
        printf("## 1: result: FAIL\n");
    }
    printf("## 1: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## 1: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
