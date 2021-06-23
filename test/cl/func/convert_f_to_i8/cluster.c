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
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* f_stm_l1;
RT_CL_DATA static int8_t* i_acq_l1;
RT_CL_DATA static int8_t* i_exp_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    //float factor = *((float*)(&mul_factor));
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    func_convert_f_to_i8(f_stm_l1, i_acq_l1, N_DIM, N_DIM, STRIDE, mul_factor);

    rt_perf_stop(perf);

    int max_diff = 0.f;

    for (int i = 0; i < N_DIM; i++) {
        for (int j = 0; j < N_DIM; j++) {
            int diff = i_acq_l1[i * STRIDE + j] - i_exp_l1[i * STRIDE + j];
            if (diff < 0) {
                diff = -diff;
            }
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > 0) {
                printf("acq: %d, exp: %d\n", i_acq_l1[i * STRIDE + j], i_exp_l1[i * STRIDE + j]);
            }
        }
    }

    if (max_diff > 0) {
        printf("## 1: abs_err: %d\n", max_diff);
        return 1;
    }

    return 0;
}

void cluster_entry(void* arg) {

    // allocate memory
    f_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    i_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * N_DIM * STRIDE);
    i_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * N_DIM * STRIDE);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)f_stm,     (unsigned int)f_stm_l1,     sizeof(f_stm),     RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)i_exp,     (unsigned int)i_exp_l1,     sizeof(i_exp),     RT_DMA_DIR_EXT2LOC, 1, &copy);
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
