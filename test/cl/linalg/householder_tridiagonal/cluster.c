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


int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_householder_tridiagonal(a_stm_l1, q_acq_l1, N_DIM, workspace_l1);

    rt_perf_stop(perf);

    float max_abs_diff = 0.f;

    for (int _i = 0; _i < N_DIM; _i++) {
        for (int _j = 0; _j < N_DIM; _j++) {
            int _idx = _i * N_DIM + _j;
            float t_abs_diff = abs_diff(a_stm_l1[_idx], t_exp_l1[_idx]);
            float q_abs_diff = abs_diff(q_acq_l1[_idx], q_exp_l1[_idx]);
            max_abs_diff = insn_fmax(max_abs_diff, t_abs_diff);
            max_abs_diff = insn_fmax(max_abs_diff, q_abs_diff);
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
    workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * (2 * N_DIM + 1));

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
