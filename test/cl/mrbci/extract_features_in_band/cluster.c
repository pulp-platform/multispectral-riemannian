#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/mrbci/mrbci.h"
#include "test_stimuli.h"

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    // allocate output
    int8_t* y_acq = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    mrbci_extract_features_in_band(x_stm, FREQ_IDX, y_acq);

    rt_perf_stop(perf);

    int error = 0;
    for (int k = 0; k < MRBCI_HALF_DIAG_FEATURES; k++) {
        if (y_acq[k] != y_exp[k]) {
            error = 1;
            printf("error at: %d: diff=%d, acq=%d, exp=%d\n", k, y_acq[k] - y_exp[k], y_acq[k], y_exp[k]);
        }
    }

    rt_free(RT_ALLOC_CL_DATA, y_acq, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

    return error;
}

void cluster_entry(void* arg) {

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
