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

RT_CL_DATA static uint32_t* a_vec_l1;
RT_CL_DATA static uint32_t* b_vec_l1;
RT_CL_DATA static uint32_t* b_exp_vec_l1;
RT_CL_DATA static func_sos_filt_2S_params_t* filt_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    func_copy_mat(a_vec_l1, b_vec_l1, N_DIM, M_DIM, STRIDE_A, STRIDE_B);

    rt_perf_stop(perf);

    int error = 0;
    for (int i = 0; i < N_DIM; i++) {
        for (int j = 0; j < STRIDE_B; j++) {
            if (b_vec_l1[i * STRIDE_B + j] != b_exp_vec_l1[i * STRIDE_B + j]) {
                error = 1;
                printf("error of B at: %d, %d: acq=%u, exp=%u\n", i, j, b_vec_l1[i * STRIDE_B + j], b_exp_vec_l1[i * STRIDE_B + j]);
            }
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // allocate memory
    a_vec_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(uint32_t) * N_DIM * STRIDE_A);
    b_vec_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(uint32_t) * N_DIM * STRIDE_B);
    b_exp_vec_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(uint32_t) * N_DIM * STRIDE_B);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_vec,     (unsigned int)a_vec_l1,     sizeof(a_vec),     RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)b_vec,     (unsigned int)b_vec_l1,     sizeof(b_vec),     RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)b_exp_vec, (unsigned int)b_exp_vec_l1, sizeof(b_exp_vec), RT_DMA_DIR_EXT2LOC, 1, &copy);
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
