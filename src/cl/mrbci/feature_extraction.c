/**
 * @file mrbci.h
 * @author Tibor Schneider
 * @date 2020/02/28
 * @brief This file contains the implementation for extracting the features
 */

#include "rt/rt_api.h"
#include "mrbci.h"

#ifndef HOUSEHOLDER_SLOW
#define _LOGM_WORKSPACE_SIZE (MRBCI_C * (2 * MRBCI_C + 2))
#else //HOUSEHOLDER_SLOW
#define _LOGM_WORKSPACE_SIZE (MRBCI_C * (3 * MRBCI_C + 2))
#endif

/**
 * @brief compute all features from the input
 *
 * @warning p_in and p_out is not allowed to be in L1, it must reside in L2 memory.
 * 
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param p_out Pointer to the entire result vector (of shape MRBCI_SVM_NUM_FEATURES) in L2.
 */
void mrbci_extract_features(const int8_t* p_in,
                            int8_t* p_out) {

    rt_dma_copy_t _copy;

    // allocate output memory in L1
    int8_t* _p_out_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);
    int8_t* _p_out_l1_iter = _p_out_l1;

    // repeat the same procedure for all frequencies
    for (unsigned int freq_idx = 0; freq_idx < MRBCI_NUM_FREQ; freq_idx++) {

        mrbci_extract_features_in_band(p_in, freq_idx, _p_out_l1_iter);
        _p_out_l1_iter += MRBCI_HALF_DIAG_FEATURES;

    }

    // copy back the result
    rt_dma_memcpy((unsigned int)p_out,
                  (unsigned int)_p_out_l1,
                  MRBCI_SVM_NUM_FEATURES,
                  RT_DMA_DIR_LOC2EXT, 0, &_copy);
    rt_dma_wait(&_copy);

    // free the allocated data
    rt_free(RT_ALLOC_CL_DATA, _p_out_l1, sizeof(int8_t) * MRBCI_SVM_NUM_FEATURES_ALIGN);

}

/**
 * @brief compute the features from the input for a given frequency band
 *
 * @warning p_in is not allowed to be in L1, it must reside in L2 memory. p_out however must be in L1.
 * 
 * @param p_in Pointer to the input matrix of shape [C, T], aligned to [C, T_ALIGN] in L2.
 * @param freq_idx Frequency id, 0 <= freq_idx < N_FREQ
 * @param p_out Pointer to the entire result vector (of shape MRBCI_HALF_DIAG_NUM_FEATURES) in L1.
 */
void mrbci_extract_features_in_band(const int8_t* p_in,
                                    unsigned int freq_idx,
                                    int8_t* p_out) {

    rt_dma_copy_t _copy;

    // allocate input data
    int8_t* _p_input_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // copy in the data
    rt_dma_memcpy((unsigned int)p_in,
                  (unsigned int)_p_input_l1,
                  sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN,
                  RT_DMA_DIR_EXT2LOC, 0, &_copy);

    // allocate the space for the filtered data
    int8_t* _p_filter_data_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // fill the aligned elements with 0 to make sure that the covmat is computed correctly
    unsigned int _last_4_columns = MRBCI_T_ALIGN - 4;
    for (unsigned int _ch = 0; _ch < MRBCI_C; _ch++) {
        *((int32_t*)(_p_filter_data_l1 + MRBCI_T_ALIGN * _ch + _last_4_columns)) = 0;
    }

    // wait until the input data is copied
    rt_dma_wait(&_copy);

    // apply the filtering
    mrbci_filter(_p_input_l1, freq_idx, _p_filter_data_l1);

    // free up the input data
    rt_free(RT_ALLOC_CL_DATA, _p_input_l1, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // allocate the space for the covariance matrix
    int16_t* _p_covmat_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int16_t) * MRBCI_C * MRBCI_C);

    // compute the covariance matrix
    mrbci_covmat(_p_filter_data_l1, freq_idx, _p_covmat_l1);

    // free the filtered data
    rt_free(RT_ALLOC_CL_DATA, _p_filter_data_l1, sizeof(int8_t) * MRBCI_C * MRBCI_T_ALIGN);

    // allocate whitening output and workspace for whitening
    int32_t* _p_whitened_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);
    int32_t* _p_whitening_workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // compute whitening
    mrbci_whitening(_p_covmat_l1, freq_idx, _p_whitened_l1, _p_whitening_workspace_l1);

    // free the covariance matrix data
    rt_free(RT_ALLOC_CL_DATA, _p_covmat_l1, sizeof(int16_t) * MRBCI_C * MRBCI_C);

    // free workspace for whitening
    rt_free(RT_ALLOC_CL_DATA, _p_whitening_workspace_l1, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // allocate result of matrix logarithm and workspace for logm
    int8_t* _p_logm_out_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN);
    float* _p_logm_workspace_l1 = rt_alloc(RT_ALLOC_CL_DATA, sizeof(float) * _LOGM_WORKSPACE_SIZE);

    // compute matrix logarithm
    mrbci_logm(_p_whitened_l1, freq_idx, _p_logm_out_l1, _p_logm_workspace_l1);

    // free whitening output
    rt_free(RT_ALLOC_CL_DATA, _p_whitened_l1, sizeof(int32_t) * MRBCI_C * MRBCI_C);

    // free logm workspace
    rt_free(RT_ALLOC_CL_DATA, _p_logm_workspace_l1, sizeof(float) * _LOGM_WORKSPACE_SIZE);

    // compute hald diagonalization
    mrbci_half_diag(_p_logm_out_l1, p_out);

    // free result of matrix logarithm
    rt_free(RT_ALLOC_CL_DATA, _p_logm_out_l1, sizeof(int8_t) * MRBCI_C * MRBCI_C_ALIGN);

}
