/**
 * @file mrbci.h
 * @author Tibor Schneider
 * @date 2020/02/20
 * @brief This file contains the definitions for all blocks in the mrbci
 */

#ifndef __CL_MRBCI_H__
#define __CL_MRBCI_H__

#include "rt/rt_api.h"
#include "mrbci_params.h"

/**
 * All arrays that cannot be determined statically
 */

const float* mrbci_logm_dequant;
float mrbci_logm_requant;

/**
 * @brief initializes mrbci
 */
void mrbci_init();

/**
 * @brief Apply the FIR filter for a given frequency
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input data of shape [C, T], aligned to [C, T_ALIGN]
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, T], aligned to [C, T_ALIGN]
 */
void mrbci_filter(const int8_t* p_in,
                  unsigned int freq_idx,
                  int8_t* p_out);

/**
 * @brief Apply compute the regularized covariance matrix of the filtered data
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to filtered input data of shape [C, T], aligned to [C, T_ALIGN]
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C], aligned to [C, C] (assuming C is even)
 */
void mrbci_covmat(const int8_t* p_in,
                  unsigned int freq_idx,
                  int16_t* p_out);

/**
 * @brief Apply the whitening transform (two matrix multiplications)
 *
 * @warning p_in, p_workspace and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C], aligned to [C, C] (assuming C is even)
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C]
 * @param p_workspace Pointer to temporary workspace data of shape [C, C]
 */
void mrbci_whitening(const int16_t* p_in,
                     unsigned int freq_idx,
                     int32_t* p_out,
                     int32_t* p_workspace);

/**
 * @brief Apply the Matrix Logarithm
 *
 * @warning p_in, p_workspace and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C], is destroyed in this function
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to output data of shape [C, C], aligned to [C, C_ALIGN]
 * @param p_workspace Pointer to temporary workspace data of shape [C, 3*C + 1]
 */
void mrbci_logm(int32_t* p_in,
                unsigned int freq_idx,
                int8_t* p_out,
                float* p_workspace);

#endif//__CL_MRBCI_H__
