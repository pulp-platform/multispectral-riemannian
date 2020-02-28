/**
 * @file filter.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the filtering
 */

#include "mrbci.h"
#include "mrbci_params.h"
#include "../func/functional.h"

/**
 * @brief Apply the FIR filter for a given frequency
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input data of shape [C, T], aligned to [C, T_ALIGN]
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, T], aligned to [C, T_ALIGN]
 */
void mrbci_filter(int8_t* p_in,
                  unsigned int freq_idx,
                  int8_t* p_out) {

    for (int _ch = 0; _ch < MRBCI_C; _ch++) {
        func_sos_filt_2S(p_in + _ch * MRBCI_T_ALIGN,
                         MRBCI_T,
                         &(mrbci_filter_params[freq_idx]),
                         p_out + _ch * MRBCI_T_ALIGN);
    }

}
