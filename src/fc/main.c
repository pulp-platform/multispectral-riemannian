/*
 * Main for Fabric Controller
 */

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
#include "../cl/cluster.h"

/** 
 * \brief Fabric main
 */
int main(void)
{

  // change the clock frequency

  //*(uint32_t *)0x1A100014=0xd0885f5e;		//200 MHz on FC
  //*(uint32_t *)0x1A100024=0xd088bebc;		//400 MHz on CL

  // 100MHz
  int freq = 160000000;

  rt_freq_set(RT_FREQ_DOMAIN_FC, 60000000);
  rt_freq_set(RT_FREQ_DOMAIN_CL, freq);

  // change the voltage
  //rt_voltage_force(RT_VOLTAGE_DOMAIN_MEMORY,600,NULL);

#ifdef POWER

    printf("fc::main::main (Power Measurements)\n");

    while(1) {

        // mount the cluster, and wait until the cluster is mounted
        rt_cluster_mount(1, 0, 0, NULL);

        // call the cluster entry point, and wait unitl it is finished
        rt_cluster_call(NULL, 0, cluster_entry, NULL, NULL, 0, 0, 0, NULL);

        // unmount the cluster
        rt_cluster_mount(0, 0, 0, NULL);

        // wait for 1 second
        rt_time_wait_us(1000000);

    }

#else//POWER

    printf("fc::main::main\n");

    // mount the cluster, and wait until the cluster is mounted
    rt_cluster_mount(1, 0, 0, NULL);

    // call the cluster entry point, and wait unitl it is finished
    rt_cluster_call(NULL, 0, cluster_entry, NULL, NULL, 0, 0, 0, NULL);

    // unmount the cluster
    rt_cluster_mount(0, 0, 0, NULL);

#endif//POWER


    return 0;
}
