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
    printf("fc::main::main\n");

    // mount the cluster, and wait until the cluster is mounted
    rt_cluster_mount(1, 0, 0, NULL);

    // call the cluster entry point, and wait unitl it is finished
    rt_cluster_call(NULL, 0, cluster_entry, NULL, NULL, 0, 0, 0, NULL);

    // unmount the cluster
    rt_cluster_mount(0, 0, 0, NULL);

    return 0;
}
