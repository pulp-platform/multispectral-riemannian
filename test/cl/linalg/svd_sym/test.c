#include "rt/rt_api.h"
#include "cluster.h"

int main() {
    // mount the cluster
    rt_cluster_mount(1, 0, 0, NULL);

    // call the cluster entry
    rt_cluster_call(NULL, 0, cluster_entry, NULL, NULL, 0, 0, 0, NULL);
    
    // unmount the cluster entry
    rt_cluster_mount(0, 0, 0, NULL);
}
