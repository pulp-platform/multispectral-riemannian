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
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* q_acq_l1;
RT_CL_DATA static float* t_exp_l1;
RT_CL_DATA static float* q_exp_l1;
RT_CL_DATA static float* workspace_l1;


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

#ifdef PARALLEL

#ifndef NUM_WORKERS
#define NUM_WORKERS 8
#endif//NUM_WORKERS

// forward declaration
void _linalg_kernel_householder_tridiagonal(unsigned int core_id, float* p_a, float* p_q, unsigned int N, float* p_workspace);

void bench_kernel(void* args) {
    unsigned int core_id = rt_core_id();
    _linalg_kernel_householder_tridiagonal(core_id, a_stm_l1, q_acq_l1, N_DIM, workspace_l1);
}
#endif//PARALLEL


int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

#ifdef PARALLEL
    rt_team_fork(NUM_WORKERS, bench_kernel, NULL);
#else//PARALLEL
    linalg_householder_tridiagonal(a_stm_l1, q_acq_l1, N_DIM, workspace_l1);
#endif//PARALLEL

    rt_perf_stop(perf);

    float max_abs_diff = 0.f;

    for (int _i = 0; _i < N_DIM; _i++) {
        for (int _j = _i; _j < N_DIM; _j++) {
            int _idx = _i * N_DIM + _j;
            float t_abs_diff = abs_diff(a_stm_l1[_idx], t_exp_l1[_idx]);
            float q_abs_diff = abs_diff(q_acq_l1[_idx], q_exp_l1[_idx]);
            max_abs_diff = insn_fmax(max_abs_diff, t_abs_diff);
            max_abs_diff = insn_fmax(max_abs_diff, q_abs_diff);
            if (t_abs_diff > EPSILON) {
                printf("Error at T: i=%d, j=%d, diff=%.2e\n", _i, _j, t_abs_diff);
            }
            if (q_abs_diff > EPSILON) {
                printf("Error at Q: i=%d, j=%d, diff=%.2e\n", _i, _j, q_abs_diff);
            }
        }
    }

    if (max_abs_diff > EPSILON) {
        printf("## 1: abs_err: %.2e\n", max_abs_diff);
        return 1;
    }

    return 0;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    // allocate memory
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    q_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    t_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    q_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * (2 * N_DIM + 2));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)t_exp, (unsigned int)t_exp_l1, sizeof(t_exp), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)q_exp, (unsigned int)q_exp_l1, sizeof(q_exp), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_wait(&copy);

    // prepare l_acq and r_acq
    linalg_fill_I(q_acq_l1, N_DIM);

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
