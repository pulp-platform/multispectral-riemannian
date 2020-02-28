#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/mrbci/mrbci.h"
#include "test_stimuli.h"

RT_CL_DATA static int8_t* x_stm_l1;
RT_CL_DATA static int8_t* y_acq_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    y_acq_l1 += (FREQ_IDX % 4); // make sure to sometimes use misaligned positions
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    mrbci_half_diag(x_stm_l1, y_acq_l1);

    rt_perf_stop(perf);

    int error = 0;
    for (int i = 0; i < MRBCI_HALF_DIAG_FEATURES; i++) {
        if (y_acq_l1[i] != y_exp[i]) {
            error = 1;
            printf("error at: %d: acq=%d, exp=%d\n", i, y_acq_l1[i], y_exp[i]);
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // allocate memory
    x_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN);
    y_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * 400);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)x_stm, (unsigned int)x_stm_l1, sizeof(x_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    mrbci_init();

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
