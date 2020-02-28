/**
 * @file mrbci.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the definitions for all blocks in the mrbci
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
