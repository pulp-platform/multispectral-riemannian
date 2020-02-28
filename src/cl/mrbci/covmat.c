/**
 * @file covmat.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the Covariance Matrix Block
 */

#include "mrbci.h"
#include "../func/functional.h"

/**
 * @brief Apply the FIR filter for a given frequency
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to filtered input data of shape [C, T], aligned to [C, T_ALIGN]
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C], aligned to [C, C] (assuming C is even)
 */
void mrbci_covmat(const int8_t* p_in,
                  unsigned int freq_idx,
                  int16_t* p_out) {

    int32_t rho = mrbci_covmat_rho[freq_idx];
    unsigned int y_shift = mrbci_covmat_shift[freq_idx];

    func_covmat_reg(p_in, rho, MRBCI_T_ALIGN, MRBCI_C, MRBCI_C, y_shift, p_out);

}
