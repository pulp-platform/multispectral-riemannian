#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* v_stm_l1;
RT_CL_DATA static float* w_stm_l1;
RT_CL_DATA static float* y_exp_l1;


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

    float _c = *((float*)c_stm);

    printf("\n\nA before:\n");
    linalg_print_mat_f(a_stm_l1, N_DIM, N_DIM, N_DIM);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_householder_update_step(a_stm_l1, v_stm_l1, w_stm_l1, _c, N_DIM, KP1);

    rt_perf_stop(perf);

    printf("\n\nA after:\n");
    linalg_print_mat_f(a_stm_l1, N_DIM, N_DIM, N_DIM);

    printf("\n\nExpected\n");
    linalg_print_mat_f(y_exp_l1, N_DIM, N_DIM, N_DIM);

    float max_abs_diff = 0.f;

    for (int _i = 0; _i < N_DIM; _i++) {
        for (int _j = 0; _j < N_DIM; _j++) {
            int _idx = _i * N_DIM + _j;
            float a_abs_diff = abs_diff(a_stm_l1[_idx], y_exp_l1[_idx]);
            max_abs_diff = insn_fmax(max_abs_diff, a_abs_diff);
            if (a_abs_diff > EPSILON) {
                printf("Error at i=%d, j=%d, exp=%.2e, acq=%.2e\n", _i, _j, y_exp_l1[_idx], a_stm_l1[_idx]);
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
    v_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM);
    w_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM);
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)v_stm, (unsigned int)v_stm_l1, sizeof(v_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)w_stm, (unsigned int)w_stm_l1, sizeof(w_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
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