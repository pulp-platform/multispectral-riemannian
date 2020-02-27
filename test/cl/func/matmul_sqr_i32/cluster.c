#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/func/functional.h"
#include "test_stimuli.h"

RT_CL_DATA static int32_t* a_stm_l1;
RT_CL_DATA static int32_t* b_stm_l1;
RT_CL_DATA static int32_t* y_acq_l1;
RT_CL_DATA static int32_t* y_exp_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    func_matmul_sqr_i32(a_stm_l1, b_stm_l1, N_DIM, y_acq_l1);

    rt_perf_stop(perf);

    int error = 0;
    for (int i = 0; i < N_DIM; i++) {
        for (int j = 0; j < N_DIM; j++) {
            if (y_acq_l1[i * N_DIM + j] != y_exp_l1[i * N_DIM + j]) {
                error = 1;
                printf("error at: %d, %d: acq=%d, exp=%d\n", i, j, y_acq_l1[i * N_DIM + j], y_exp_l1[i * N_DIM + j]);
            }
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // allocate memory
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * N_DIM * N_DIM);
    b_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * N_DIM * N_DIM);
    y_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * N_DIM * N_DIM);
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * N_DIM * N_DIM);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)b_stm, (unsigned int)b_stm_l1, sizeof(b_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)y_exp, (unsigned int)y_exp_l1, sizeof(y_exp), RT_DMA_DIR_EXT2LOC, 1, &copy);
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
