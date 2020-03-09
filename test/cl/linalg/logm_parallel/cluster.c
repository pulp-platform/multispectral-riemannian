#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* y_exp_l1;
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

    linalg_logm_parallel(a_stm_l1, N_DIM, workspace_l1);

    rt_perf_stop(perf);

    float max_rel_diff = 0.f;

    for (int _i = 0; _i < N_DIM; _i++) {
        for (int _j = 0; _j < N_DIM; _j++) {
            int _idx = _i * N_DIM + _j;
            float a_rel_diff = rel_diff(a_stm_l1[_idx], y_exp_l1[_idx]);
            max_rel_diff = insn_fmax(max_rel_diff, a_rel_diff);
        }
    }

    if (max_rel_diff > EPSILON) {
        printf("## 1: rel_err: %.2e\n", max_rel_diff);
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
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * (N_DIM * 3 + 2));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
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

    rt_free(RT_ALLOC_CL_DATA, a_stm_l1, sizeof(float) * N_DIM * N_DIM);
    rt_free(RT_ALLOC_CL_DATA, y_exp_l1, sizeof(float) * N_DIM * N_DIM);
    rt_free(RT_ALLOC_CL_DATA, workspace_l1, sizeof(float) * N_DIM * (N_DIM * 3 + 1));

}