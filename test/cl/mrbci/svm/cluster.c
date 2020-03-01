#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/mrbci/mrbci.h"
#include "test_stimuli.h"

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    // allocate output
    int32_t* y_acq = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int32_t) * MRBCI_NUM_CLASS);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    mrbci_svm(x_stm, y_acq);

    rt_perf_stop(perf);

    int error = 0;
    for (int k = 0; k < MRBCI_NUM_CLASS; k++) {
        if (y_acq[k] != y_exp[k]) {
            error = 1;
            printf("error at: %d: acq=%d, exp=%d\n", k, y_acq[k], y_exp[k]);
        }
    }

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
