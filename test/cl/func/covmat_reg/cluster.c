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
#include "../../../../src/cl/func/functional.h"
#include "test_stimuli.h"

RT_CL_DATA static int8_t* x_stm_l1;
RT_CL_DATA static int16_t* y_acq_l1;
RT_CL_DATA static int16_t* y_exp_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    func_covmat_reg(x_stm_l1, rho, M_ALIGN, N_DIM, N_DIM, y_shift, y_acq_l1);

    rt_perf_stop(perf);

    int error = 0;
    for (int i = 0; i < N_DIM; i++) {
        for (int j = 0; j < N_DIM; j++) {
            if (y_acq_l1[i * N_DIM + j] != y_exp_l1[i * N_DIM + j]) {
                error = 1;
                printf("error at: %d, %d: acq=%d, exp=%d\n", i, j, y_acq_l1[i * N_DIM + j], y_exp_l1[i * N_DIM + j]);
            }
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // allocate memory
    x_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * N_DIM * M_ALIGN);
    y_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int16_t) * N_DIM * N_DIM);
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int16_t) * N_DIM * N_DIM);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)x_stm, (unsigned int)x_stm_l1, sizeof(x_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)y_exp, (unsigned int)y_exp_l1, sizeof(y_exp), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_wait(&copy);

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

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
