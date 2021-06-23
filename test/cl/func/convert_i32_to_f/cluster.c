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

RT_CL_DATA static int32_t* i_stm_l1;
RT_CL_DATA static float* f_acq_l1;
RT_CL_DATA static float* f_exp_l1;

float rel_diff(float exp, float acq) {
    if (exp == acq) {
        return 0.f;
    }
    float abs_diff = insn_fabs(exp - acq);
    float rel_diff = abs_diff / fabs(exp);
    return rel_diff;
}

float abs_diff(float exp, float acq) {
    if (exp == acq) {
        return 0.f;
    }
    float abs_diff = insn_fabs(exp - acq);
    return abs_diff;
}

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    float factor = *((float*)(&mul_factor));
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    func_convert_i32_to_f(i_stm_l1, f_acq_l1, N_DIM * N_DIM, factor);

    rt_perf_stop(perf);

    float max_abs_diff = 0.f;

    for (int i = 0; i < N_DIM; i++) {
        for (int j = 0; j < N_DIM; j++) {
            float f_abs_diff = abs_diff(f_exp_l1[i * N_DIM + j], f_acq_l1[i * N_DIM + j]);
            max_abs_diff = insn_fmax(max_abs_diff, f_abs_diff);
            printf("abs_err: %.2e\n", f_abs_diff);
        }
    }

    if (max_abs_diff > EPSILON) {
        printf("## 1: abs_err: %.2e\n", max_abs_diff);
        return 1;
    }

    return 0;
}

void cluster_entry(void* arg) {

    // allocate memory
    i_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * N_DIM * N_DIM);
    f_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    f_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)i_stm,     (unsigned int)i_stm_l1,     sizeof(i_stm),     RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)f_exp,     (unsigned int)f_exp_l1,     sizeof(f_exp),     RT_DMA_DIR_EXT2LOC, 1, &copy);
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
