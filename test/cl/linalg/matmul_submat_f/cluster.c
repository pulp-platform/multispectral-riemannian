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
#include "../../../../src/cl/linalg/linalg.h"
#include "../../../../src/cl/func/functional.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* b_stm_l1;
RT_CL_DATA static float* y_acq_l1;
RT_CL_DATA static float* y_exp_l1;

float rel_diff(float exp, float acq) {
    if (exp == acq) {
        return 0.f;
    }
    float abs_diff = fabs(exp - acq);
    float rel_diff = abs_diff / fabs(exp);
    return rel_diff;
}

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    unsigned int k = MAT_DIM - SUBMAT_DIM;

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    if (RIGHT_SPECIAL) {

        func_copy_mat((uint32_t*)a_stm_l1, (uint32_t*)y_acq_l1, MAT_DIM, k, MAT_DIM, MAT_DIM);
        linalg_matmul_stride_f(a_stm_l1 + k,
                               b_stm_l1 + k * (MAT_DIM + 1),
                               MAT_DIM, SUBMAT_DIM, SUBMAT_DIM,
                               MAT_DIM, MAT_DIM, MAT_DIM,
                               y_acq_l1 + k);

    } else {

        func_copy_mat((uint32_t*)b_stm_l1, (uint32_t*)y_acq_l1, SUBMAT_DIM, MAT_DIM, MAT_DIM, MAT_DIM);
        linalg_matmul_stride_f(a_stm_l1 + k * (MAT_DIM + 1),
                               b_stm_l1 + k * MAT_DIM,
                               SUBMAT_DIM, SUBMAT_DIM, MAT_DIM,
                               MAT_DIM, MAT_DIM, MAT_DIM,
                               y_acq_l1 + k * MAT_DIM);

    }
    rt_perf_stop(perf);

    float max_rel_diff = 0;

    int error = 0;
    for (int _m = 0; _m < MAT_DIM; _m++) {
        for (int _o = 0; _o < MAT_DIM; _o++) {
            int _idx = _m * MAT_DIM + _o;
            float _rel_diff = rel_diff(y_exp_l1[_idx], y_acq_l1[_idx]);
            max_rel_diff = insn_fmax(max_rel_diff, _rel_diff);
            if (_rel_diff > EPSILON) {
                // printf("error at M=%d, O=%d: diff=%.2e\n", _m, _o, _rel_diff);
                error = 1;
            }
        }
    }

    if (error == 1) {
        printf("## 1: rel_err: %.2e\n", max_rel_diff);
    }

    return error;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    // allocate memory
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * MAT_DIM * MAT_DIM);
    b_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * MAT_DIM * MAT_DIM);
    y_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * MAT_DIM * MAT_DIM);
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * MAT_DIM * MAT_DIM);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)b_stm, (unsigned int)b_stm_l1, sizeof(b_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)y_exp, (unsigned int)y_exp_l1, sizeof(y_exp), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_wait(&copy);

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
