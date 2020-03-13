#include "rt/rt_api.h"
#include "stdio.h"
#include "cluster.h"
#include "mrbci/mrbci.h"
#include "input.h"

/** 
 * \brief Cluster entry point (main)
 */
void cluster_entry(void *arg)
{
    // allocate output memory
    int32_t * _p_output = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(int32_t) * MRBCI_NUM_CLASS);

    // initialize the module
    mrbci_init();

    // compute the model
    mrbci_run(input_data, _p_output);

#ifndef POWER
    // print the result
    printf("Result:\n");
    for (int i = 0; i < MRBCI_NUM_CLASS; i++) {
        printf("Class %d: %d\n", i + 1, _p_output[i]);
    }
#endif//POWER

    // free memory
    rt_free(RT_ALLOC_L2_CL_DATA, (void*)_p_output, sizeof(int32_t) * MRBCI_NUM_CLASS);
}
