/**
 * @file mrbci.h
 * @author Tibor Schneider
 * @date 2020/02/28
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
 * @brief Do the entire multiscale reiamnnian BCI
 *
 * @brief p_in and p_out are not allowed to be in L1, they must reside in L2 memory.
 *
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param p_out Pointer to result of shape [MRBCI_NUM_CLASS] in L2
 */
void mrbci_run(const int8_t* p_in,
               int32_t* p_out);

/**
 * @brief compute all features from the input
 *
 * @warning p_in and p_out is not allowed to be in L1, it must reside in L2 memory.
 * 
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param p_out Pointer to the entire result vector (of shape MRBCI_SVM_NUM_FEATURES) in L2.
 */
void mrbci_extract_features(const int8_t* p_in,
                            int8_t* p_out);

/**
 * @brief compute SVM (matrix multiplication followed by vector addition)
 *
 * @warning p_in and p_out must reside on L2 memory, not on L1.
 *
 * @param p_in Pointer to feature map of shape [MRBCI_SVM_NUM_FEATURES] in L2 (aligned)
 * @param p_out Pointer to result of shape [MRBCI_NUM_CLASS] in L2
 */
void mrbci_svm(const int8_t* p_in,
               int32_t* p_out);

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

/**
 * @brief Convert the upper right half of the matrix into a 
 *
 * @warning p_in and p_out should be placed on L1, and be allocated
 *
 * @param p_in Pointer to input matrix of shape [C, C_ALIGN], is destroyed in this function
 * @param p_out Pointer to output data, must at least MRBCI_HALF_DIAG_FEATURES space
 */
void mrbci_half_diag(const int8_t* p_in,
                     int8_t* p_out);

#endif//__CL_MRBCI_H__
