#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_evd_2x2_t acq = linalg_evd_2x2(a_stm, b_stm, c_stm);

    rt_perf_stop(perf);

    if (fabs(cs_exp - acq.cs) > EPSILON
        || fabs(sn_exp - acq.sn) > EPSILON
        || fabs(ev1_exp - acq.ev1) > EPSILON
        || fabs(ev2_exp - acq.ev2) > EPSILON) {
        printf("cs diff: %.2e, sn diff: %.2e\n", fabs(cs_exp - acq.cs), fabs(sn_exp - acq.sn));
        printf("ev1 diff: %.2e, ev2 diff: %.2e\n", fabs(ev1_exp - acq.ev1), fabs(ev2_exp - acq.ev2));
        return 1;
    }

    return 0;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    int result;

    for (int i = 0; i < 10; i++) {
        result = do_bench(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));
    }

    // print the results
    if (result == 0) {
        printf("## 1: result: OK\n");
    } else {
        printf("## 1: result: FAIL\n");
    }
    printf("## 1: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## 1: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
