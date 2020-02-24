#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"

RT_CL_DATA static float* mat_l1;

int do_bench(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_fill_I(mat_l1, DIM);

    rt_perf_stop(perf);

    int error = 0;
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            if (i == j) {
                if (mat_l1[i + DIM * j] != 1.f) {
                    error = 1;
                }
            } else {
                if (mat_l1[i + DIM * j] != 0.f) {
                    error = 1;
                }
            }
        }
    }

    return error;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    // allocate memory
    mat_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * DIM * DIM);

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
