#include "stdio.h"
#include "rt/rt_api.h"
#include "../../../../src/cl/linalg/linalg.h"
#include "test_stimuli.h"
#include "math.h"
#include "../../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* y_acq_l1;
RT_CL_DATA static float* y_exp_l1;

float rel_diff(float exp, float acq) {
    if (exp == acq) {
        return 0.f;
    }
    float abs_diff = fabs(exp - acq);
    float rel_diff = abs_diff / fabs(exp);
    return rel_diff;
}

int do_bench_0(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);

    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_vcovmat_f(a_stm_l1 + K_REM, DIM, STRIDE, 0, y_acq_l1 + (K_REM * STRIDE + K_REM));

    rt_perf_stop(perf);

    float max_rel_diff = 0;

    int error = 0;
    for (int _i = K_REM; _i < STRIDE; _i++) {
        for (int _j = K_REM; _j < STRIDE; _j++) {
            int _idx = _i * STRIDE + _j;
            float _rel_diff = rel_diff(y_exp_l1[_idx], y_acq_l1[_idx]);
            max_rel_diff = insn_fmax(max_rel_diff, _rel_diff);
            if (_rel_diff > EPSILON) {
                printf("error at i=%d, j=%d: diff=%.2e\n", _i, _j, _rel_diff);
                error = 1;
            }
        }
    }

    if (error == 1) {
        printf("## store_all: rel_err: %.2e\n", max_rel_diff);
    }

    return error;
}

int do_bench_1(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_vcovmat_f(a_stm_l1 + K_REM, DIM, STRIDE, 1, y_acq_l1 + (K_REM * STRIDE + K_REM));

    rt_perf_stop(perf);

    float max_rel_diff = 0;

    int error = 0;
    for (int _i = K_REM; _i < STRIDE; _i++) {
        for (int _j = K_REM; _j < STRIDE; _j++) {
            if (_j >= _i) {
                int _idx = _i * STRIDE + _j;
                float _rel_diff = rel_diff(y_exp_l1[_idx], y_acq_l1[_idx]);
                max_rel_diff = insn_fmax(max_rel_diff, _rel_diff);
                if (_rel_diff > EPSILON) {
                    // printf("error at M=%d, O=%d: diff=%.2e\n", _m, _o, _rel_diff);
                    error = 1;
                }
            }
        }
    }

    if (error == 1) {
        printf("## store_upper_right: rel_err: %.2e\n", max_rel_diff);
    }

    return error;
}

int do_bench_2(rt_perf_t* perf, int events) {
    //setup performance measurement
    rt_perf_conf(perf, events);
    
    // start performance measurement
    rt_perf_reset(perf);
    rt_perf_start(perf);

    linalg_vcovmat_f(a_stm_l1 + K_REM, DIM, STRIDE, 2, y_acq_l1 + (K_REM * STRIDE + K_REM));

    rt_perf_stop(perf);

    float max_rel_diff = 0;

    int error = 0;
    for (int _i = K_REM; _i < STRIDE; _i++) {
        for (int _j = K_REM; _j < STRIDE; _j++) {
            if (_i >= _j) {
                int _idx = _i * STRIDE + _j;
                float _rel_diff = rel_diff(y_exp_l1[_idx], y_acq_l1[_idx]);
                max_rel_diff = insn_fmax(max_rel_diff, _rel_diff);
                if (_rel_diff > EPSILON) {
                    // printf("error at M=%d, O=%d: diff=%.2e\n", _m, _o, _rel_diff);
                    error = 1;
                }
            }
        }
    }

    if (error == 1) {
        printf("## store_lower_left: rel_err: %.2e\n", max_rel_diff);
    }

    return error;
}

void cluster_entry(void* arg) {

    // setup performance measurement
    rt_perf_t perf;
    rt_perf_init(&perf);

    // allocate memory
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * STRIDE);
    y_acq_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * STRIDE * STRIDE);
    y_exp_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * STRIDE * STRIDE);

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)y_exp, (unsigned int)y_exp_l1, sizeof(y_exp), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_wait(&copy);

    int result;

    result = do_bench_0(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## store_all: result: OK\n");
    } else {
        printf("## store_all: result: FAIL\n");
    }
    printf("## store_all: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## store_all: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));

    // reset
    linalg_fill_I(y_acq_l1, DIM);

    result = do_bench_1(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## store_upper_right: result: OK\n");
    } else {
        printf("## store_upper_right: result: FAIL\n");
    }
    printf("## store_upper_right: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## store_upper_right: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));

    // reset
    linalg_fill_I(y_acq_l1, DIM);

    result = do_bench_2(&perf, (1<<RT_PERF_CYCLES | 1<<RT_PERF_INSTR));

    // print the results
    if (result == 0) {
        printf("## store_lower_left: result: OK\n");
    } else {
        printf("## store_lower_left: result: FAIL\n");
    }
    printf("## store_lower_left: cycles: %d\n", rt_perf_read(RT_PERF_CYCLES));
    printf("## store_lower_left: instructions: %d\n", rt_perf_read(RT_PERF_INSTR));
}
