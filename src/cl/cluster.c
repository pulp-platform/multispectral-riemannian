#include "rt/rt_api.h"
#include "stdio.h"
#include "cluster.h"

/** 
 * \brief Cluster entry point (main)
 */
void cluster_entry(void *arg)
{
    printf("cl::cluster::cluster_entry (core %d)\n", rt_core_id());
}
