#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* b_stm_l1;

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

    float _y_acq;
    float _y_exp = *(float*)y_exp;

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    _y_acq = linalg_vec_innerprod_f(a_stm_l1, b_stm_l1, N_DIM);

    rt_perf_stop(perf);

    float _rel_diff = rel_diff(_y_exp, _y_acq);

    if (_rel_diff > EPSILON) {
        printf("## 1: rel_err: %.2e\n", _rel_diff);
        return 1;
    }

    return 0;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    // allocate memory
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM);
    b_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * N_DIM);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)b_stm, (unsigned int)b_stm_l1, sizeof(b_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
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
