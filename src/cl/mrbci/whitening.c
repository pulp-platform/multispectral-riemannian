/**
 * @file whitening.c
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the implementation for the Whitening Block
 */

#include "mrbci.h"
#include "../func/functional.h"

/**
 * @brief Apply the whitening transform (two matrix multiplications)
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C], aligned to [C, C] (assuming C is even)
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C]
 * @param p_workspace Pointer to temporary workspace data of shape [C, C]
 */
void mrbci_whitening(const int16_t* p_in,
                     unsigned int freq_idx,
                     int32_t* p_out,
                     int32_t* p_workspace) {

    const int16_t* c_ref_invsqrtm_i16 = mrbci_c_ref_invsqrtm_i16 + (MRBCI_C * MRBCI_C) * freq_idx;
    const int32_t* c_ref_invsqrtm_i32 = mrbci_c_ref_invsqrtm_i32 + (MRBCI_C * MRBCI_C) * freq_idx;

    func_matmul_sqr_i16(p_in, c_ref_invsqrtm_i16, MRBCI_C, p_workspace);
    func_matmul_sqr_i32(c_ref_invsqrtm_i32, p_workspace, MRBCI_C, p_out);

}
