/**
 * @file mrbci.c
 * @author Tibor Schneider
 * @date 2020/02/28
 * @brief This file contains the init and the main run function for the MRBCI
 */

#include "rt/rt_api.h"
#include "mrbci.h"
#include "mrbci_params.h"
#include "../func/functional.h"

/**
 * @brief initializes mrbci
 */
void mrbci_init() {
    mrbci_logm_dequant = (float*)mrbci_logm_dequant_i;
    mrbci_logm_requant = *(float*)(&mrbci_logm_requant_i);
}


/**
 * @brief Do the entire multiscale reiamnnian BCI
 *
 * @brief p_in and p_out are not allowed to be in L1, they must reside in L2 memory.
 *
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param p_out Pointer to result of shape [MRBCI_NUM_CLASS] in L2
 */
void mrbci_run(const int8_t* p_in,
               int32_t* p_out) {

    // Allocate the output vector on L2
    int8_t* _feature_map = rt_alloc(RT_ALLOC_FC_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

    // compute all the features
    mrbci_extract_features(p_in, _feature_map);

    // compute the output
    mrbci_svm(_feature_map, p_out);

    // free up the allocated memory
    rt_free(RT_ALLOC_FC_DATA, _feature_map, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
}
