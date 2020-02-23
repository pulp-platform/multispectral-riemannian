#include "stdio.h"
#include "rt/rt_api.h"
#include "test_stimuli.h"
#include "../../../src/cl/insn.h"

RT_CL_DATA static float* stm_l1;

void cluster_entry(void* arg) {

    // allocate memory
    stm_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(stm));

    // copy memory
    rt_dma_copy_t copy;
    rt_dma_memcpy((unsigned int)stm, (unsigned int)stm_l1, sizeof(stm), RT_DMA_DIR_EXT2LOC, 0, &copy);
    rt_dma_wait(&copy);


    float res;
    uint32_t res_int;

    // do addition
    res = stm_l1[0] + stm_l1[1];
    res_int = *((uint32_t*)(&res));
    printf("## add: res: 0x%x\n", res_int);

    // do subtraction
    res = stm_l1[2] - stm_l1[3];
    res_int = *((uint32_t*)(&res));
    printf("## sub: res: 0x%x\n", res_int);

    // do multiplication
    res = stm_l1[4] * stm_l1[5];
    res_int = *((uint32_t*)(&res));
    printf("## mul: res: 0x%x\n", res_int);

    // do division
    res = stm_l1[6] / stm_l1[7];
    res_int = *((uint32_t*)(&res));
    printf("## div: res: 0x%x\n", res_int);

    // do square root
    res = insn_fsqrt(stm_l1[8]);
    res_int = *((uint32_t*)(&res));
    printf("## sqrt: res: 0x%x\n", res_int);

}
