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

RT_CL_DATA static int32_t* x_stm_l1;
RT_CL_DATA static int8_t* y_acq_l1;
RT_CL_DATA static float* workspace_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    mrbci_logm(x_stm_l1, FREQ_IDX, y_acq_l1, workspace_l1);

    rt_perf_stop(perf);

    int error = 0;
    for (int ch = 0; ch < MRBCI_C; ch++) {
        for (int chi = 0; chi < MRBCI_C; chi++) {
            if (y_acq_l1[ch * MRBCI_C_ALIGN + chi] != y_exp[ch * MRBCI_C_ALIGN + chi]) {
                error = 1;
                printf("error at: %d, %d: acq=%d, exp=%d\n", ch, chi, y_acq_l1[ch * MRBCI_C_ALIGN + chi], y_exp[ch * MRBCI_C_ALIGN + chi]);
            }
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // allocate memory
    x_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);
    y_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN);
    workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * MRBCI_C * (MRBCI_C * 3 + 2));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)x_stm, (unsigned int)x_stm_l1, sizeof(x_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

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
