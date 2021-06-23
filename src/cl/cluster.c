/*
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
