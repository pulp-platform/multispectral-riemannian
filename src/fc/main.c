/*
 * Main for Fabric Controller
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

    // 100MHz
    int freq = 100000000;

    rt_freq_set(RT_FREQ_DOMAIN_FC, freq);
    rt_freq_set(RT_FREQ_DOMAIN_CL, freq);

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
