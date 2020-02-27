#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../src/cl/insn.h"

RT_CL_DATA static float* a_stm_l1;
RT_CL_DATA static float* b_stm_l1;
RT_CL_DATA static float* c_stm_l1;
RT_CL_DATA static int32_t* d_stm_l1;

void cluster_entry(void* arg) {

    // allocate memory
    a_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(a_stm));
    b_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(b_stm));
    c_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(c_stm));
    d_stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(d_stm));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)a_stm, (unsigned int)a_stm_l1, sizeof(a_stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_memcpy((unsigned int)b_stm, (unsigned int)b_stm_l1, sizeof(b_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)c_stm, (unsigned int)c_stm_l1, sizeof(c_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_memcpy((unsigned int)d_stm, (unsigned int)d_stm_l1, sizeof(d_stm), RT_DMA_DIR_EXT2LOC, 1, &copy);
    rt_dma_wait(&copy);


    float res;
    uint32_t res_int;

    // do addition
    for (int i = 0; i < LENGTH; i++) {
        res = a_stm_l1[i] + b_stm_l1[i];
        res_int = *((uint32_t*)(&res));
        printf("## add: %d: 0x%x\n", i, res_int);
    }

    // do subtraction
    for (int i = 0; i < LENGTH; i++) {
        res = a_stm_l1[i] - b_stm_l1[i];
        res_int = *((uint32_t*)(&res));
        printf("## sub: %d: 0x%x\n", i, res_int);
    }

    // do multiplication
    for (int i = 0; i < LENGTH; i++) {
        res = a_stm_l1[i] * b_stm_l1[i];
        res_int = *((uint32_t*)(&res));
        printf("## mul: %d: 0x%x\n", i, res_int);
    }

    // do division
    for (int i = 0; i < LENGTH; i++) {
        res = a_stm_l1[i] / b_stm_l1[i];
        res_int = *((uint32_t*)(&res));
        printf("## div: %d: 0x%x\n", i, res_int);
    }

    // do square root
    for (int i = 0; i < LENGTH; i++) {
        res = insn_fsqrt(c_stm_l1[i]);
        res_int = *((uint32_t*)(&res));
        printf("## sqrt: %d: 0x%x\n", i, res_int);
    }

    // do multiply add
    for (int i = 0; i < LENGTH; i++) {
        res = insn_fmadd(a_stm_l1[i], b_stm_l1[i], c_stm_l1[i]);
        res_int = *((uint32_t*)(&res));
        printf("## madd: %d: 0x%x\n", i, res_int);
    }

    // do multiply subtract
    for (int i = 0; i < LENGTH; i++) {
        res = insn_fmsub(a_stm_l1[i], b_stm_l1[i], c_stm_l1[i]);
        res_int = *((uint32_t*)(&res));
        printf("## msub: %d: 0x%x\n", i, res_int);
    }

    // do neg multiply add
    for (int i = 0; i < LENGTH; i++) {
        res = insn_fnmadd(a_stm_l1[i], b_stm_l1[i], c_stm_l1[i]);
        res_int = *((uint32_t*)(&res));
        printf("## nmadd: %d: 0x%x\n", i, res_int);
    }

    // do neg multiply subtract
    for (int i = 0; i < LENGTH; i++) {
        res = insn_fnmsub(a_stm_l1[i], b_stm_l1[i], c_stm_l1[i]);
        res_int = *((uint32_t*)(&res));
        printf("## nmsub: %d: 0x%x\n", i, res_int);
    }

    // do float casting
    for (int i = 0; i < LENGTH; i++) {
        res = (float)d_stm_l1[i];
        res_int = *((uint32_t*)(&res));
        printf("## fcvt: %d: 0x%x\n", i, res_int);
    }

}
