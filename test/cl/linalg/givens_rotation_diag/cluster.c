#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

float abs_diff(float exp, float acq) {
    if (exp == acq) {
        return 0.f;
    }
    float abs_diff = fabs(exp - acq);
    return abs_diff;
}

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_givens_rotation_t acq = linalg_givens_rotation_diag(a_stm, b_stm, c_stm);

    rt_perf_stop(perf);

    float max_abs_diff = insn_fmax(abs_diff(cs_exp, acq.cs), abs_diff(sn_exp, acq.sn));

    printf("## 1: abs_err: %.2e\n", max_abs_diff);

    if (max_abs_diff > EPSILON) {
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
