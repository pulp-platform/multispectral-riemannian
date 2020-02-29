#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
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

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_2aat_f(a_stm_l1, N_DIM, K_DIM);

    rt_perf_stop(perf);

    float max_rel_diff = 0;

    int error = 0;
    for (int _m = K_DIM; _m < N_DIM; _m++) {
        for (int _n = _m; _n < N_DIM; _n++) {
            int _idx = _m * N_DIM + _n;
            float _rel_diff = rel_diff(y_exp_l1[_idx], a_stm_l1[_idx]);
            max_rel_diff = insn_fmax(max_rel_diff, _rel_diff);
            if (_rel_diff > EPSILON) {
                // printf("error at m=%d, n=%d: diff=%.2e\n", _m, _n, _rel_diff);
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
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM * N_DIM);

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
}
